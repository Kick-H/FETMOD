"""Elastic constant calculations and pressure-dependent scans."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.units import GPa, kg
from calorine.tools import (
    get_elastic_stiffness_tensor,
    get_force_constants,
    relax_structure,
)
from matplotlib import pyplot as plt
from pandas import DataFrame

from .calculators import build_nep_calculator
from .plotting import set_nature_style


def compute_elastic_constants_and_scan(
    atoms: Atoms,
    nep_path: str,
    supercell: Sequence[int] = (2, 2, 2),
    volsc_range: Sequence[float] = np.arange(0.88, 1.05, 0.04),
    name: str = "material",
    outdir: str | Path = ".",
    compute_vsound: bool = True,
) -> Tuple[np.ndarray, DataFrame]:
    """Compute elastic constants at equilibrium and across a pressure scan."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_nature_style()

    atoms = atoms.copy()
    atoms.calc = build_nep_calculator(nep_path)

    print(f"[{name}] Relaxing structure for elastic tensor ...")
    relax_structure(atoms, fmax=1.0e-4)
    cij = get_elastic_stiffness_tensor(atoms)
    with np.printoptions(precision=1, suppress=True):
        print(f"[{name}] C_ij (GPa-like units):")
        print(cij)

    if compute_vsound:
        print(f"[{name}] Computing phonons and simple sound velocity ...")
        phonon = get_force_constants(atoms, atoms.calc, supercell)
        phonon.run_band_structure([[[0.01, 0.0, 0.0]]], with_group_velocities=True)
        band = phonon.get_band_structure_dict()
        gv = np.linalg.norm(band["group_velocities"][0], axis=2) * 1e-10 / 1e-12
        vs = gv[0][2]
        density = np.sum(atoms.get_masses()) / atoms.get_volume() / kg / 1e-30
        c11_vs = vs**2 * density * 1e-9
        print(f"[{name}] v_sound(LA,[100]) = {vs:9.1f} m/s")
        print(f"[{name}] c11 ~ v^2 rho     = {c11_vs:9.1f} GPa")

    print(f"[{name}] Volume scan for C_ij vs pressure ...")
    data = []
    for volsc in volsc_range:
        s = atoms.copy()
        cell = s.get_cell()
        scale_len = volsc ** (1.0 / 3.0)
        cell *= scale_len
        s.set_cell(cell, scale_atoms=True)
        s.calc = build_nep_calculator(nep_path)

        relax_structure(s, constant_volume=True)
        cij_rlx = get_elastic_stiffness_tensor(s)
        pressure = -np.sum(s.get_stress()[:3]) / 3.0 / GPa
        vpa = s.get_volume() / len(s)

        data.append(
            dict(
                structure=name,
                volsc=volsc,
                pressure=pressure,
                volume_per_atom=vpa,
                c11=cij_rlx[0, 0],
                c12=cij_rlx[0, 1],
                c44=cij_rlx[3, 3],
            )
        )

    df = DataFrame(data).sort_values("pressure").reset_index(drop=True)

    csv_path = outdir / f"{name}_cij_vs_pressure.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{name}] volume-scan data saved to {csv_path}")

    fig, ax = plt.subplots()
    for col in ["c11", "c12", "c44"]:
        ax.plot(
            df["pressure"],
            df[col],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=3,
            label=col,
        )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Elastic constant (GPa)")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig_path = outdir / f"{name}_cij_vs_pressure.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[{name}] figure saved to {fig_path}")

    return cij, df
