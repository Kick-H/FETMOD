"""Lattice relaxation and lattice-constant scanning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from ase.units import GPa
from matplotlib import pyplot as plt
from pandas import DataFrame

from .calculators import build_nep_calculator
from .plotting import set_nature_style


@dataclass
class LatticeScanResult:
    """Results from a lattice constant scan."""

    df: DataFrame
    a_Emin: float
    E_min: float
    a_P0: float
    P_near0: float


def relax_cell_and_scan_a(
    atoms: Atoms,
    nep_path: str,
    scales: Sequence[float] = np.arange(0.95, 1.051, 0.001),
    fmax: float = 0.01,
    name: str = "material",
    outdir: str | Path = ".",
    make_plot: bool = True,
) -> LatticeScanResult:
    """Relax the cell and scan isotropic lattice constant variations."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_nature_style()

    atoms = atoms.copy()
    atoms.calc = build_nep_calculator(nep_path)

    print(f"[{name}] Relax cell + atoms ...")
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf, logfile=None)
    opt.run(fmax=fmax)

    a0 = atoms.cell.lengths()[0]
    p0 = -np.mean(atoms.get_stress()[:3]) / GPa
    print(f"[{name}] relaxed: a0 = {a0:.4f} Å, P0 = {p0:.3f} GPa")

    records = []
    for scale in scales:
        s = atoms.copy()
        s.set_cell(atoms.cell * scale, scale_atoms=True)
        s.calc = build_nep_calculator(nep_path)

        opt = BFGS(s, logfile=None)
        opt.run(fmax=fmax)

        a = s.cell.lengths()[0]
        energy = s.get_potential_energy() / len(s)
        pressure = -np.mean(s.get_stress()[:3]) / GPa

        records.append(
            dict(
                structure=name,
                a=a,
                energy=energy,
                pressure=pressure,
                scale=scale,
            )
        )

    df = DataFrame(records).sort_values("a").reset_index(drop=True)

    idx_Emin = df["energy"].idxmin()
    a_Emin = df.loc[idx_Emin, "a"]
    E_min = df.loc[idx_Emin, "energy"]

    idx_P0 = df["pressure"].abs().idxmin()
    a_P0 = df.loc[idx_P0, "a"]
    P_near0 = df.loc[idx_P0, "pressure"]

    print(f"[{name}] E_min at a = {a_Emin:.6f} Å, E = {E_min:.6f} eV/atom")
    print(f"[{name}] P ~ 0 at a = {a_P0:.6f} Å, P = {P_near0:.4f} GPa")

    csv_path = outdir / f"{name}_E_P_vs_a.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{name}] data saved to {csv_path}")

    if make_plot:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax_energy = plt.subplots()
        ax_p = ax_energy.twinx()

        color1 = color_cycle[0]
        color2 = color_cycle[1]

        line_e, = ax_energy.plot(
            df["a"],
            df["energy"],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=3,
            label="Energy",
            color=color1,
        )
        line_p, = ax_p.plot(
            df["a"],
            df["pressure"],
            marker="s",
            linestyle="--",
            linewidth=1.0,
            markersize=3,
            label="Pressure",
            color=color2,
        )

        ax_energy.set_xlabel(r"Lattice constant $a$ ($\mathrm{\AA}$)")
        ax_energy.set_ylabel("Energy (eV/atom)")
        ax_p.set_ylabel("Pressure (GPa)")

        lines = [line_e, line_p]
        labels = [l.get_label() for l in lines]
        ax_energy.legend(lines, labels, loc="best", frameon=False)

        plt.tight_layout()
        fig_path = outdir / f"{name}_relax.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[{name}] figure saved to {fig_path}")

    return LatticeScanResult(
        df=df,
        a_Emin=a_Emin,
        E_min=E_min,
        a_P0=a_P0,
        P_near0=P_near0,
    )
