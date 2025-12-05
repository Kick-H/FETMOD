"""Thermal conductivity calculations using phono3py's BTE workflow."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from calorine.tools import relax_structure
from matplotlib import pyplot as plt
from pandas import DataFrame
from phono3py import Phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phonopy.interface.calculator import read_crystal_structure

from .calculators import build_nep_calculator
from .plotting import set_nature_style


def _phonopy_to_ase_atoms(ph_atoms) -> Atoms:
    """Convert a Phonopy-style atoms object to ASE ``Atoms``."""
    return Atoms(
        symbols=ph_atoms.symbols,
        positions=ph_atoms.positions,
        cell=ph_atoms.cell,
        pbc=True,
    )


def compute_kappa_bte(
    prim: Atoms,
    nep_path: str,
    dim: Tuple[int, int, int] = (3, 3, 3),
    mesh: Tuple[int, int, int] = (10, 10, 10),
    T_min: int = 300,
    T_max: int = 1000,
    T_step: int = 10,
    name: str = "material",
    work_root: str | Path = ".",
) -> DataFrame:
    """Compute BTE thermal conductivity using NEP forces and phono3py."""
    work_root = Path(work_root).resolve()
    set_nature_style()

    work_dir = work_root / f"phono3py_{name}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print(f"Structure        : {name}")
    print(f"Work directory   : {work_dir}")
    print(f"Supercell dim    : {dim}")
    print(f"Mesh             : {mesh}")
    print("========================================")

    prim = prim.copy()
    prim.calc = build_nep_calculator(nep_path)
    print(f"[{name}] Relaxing primitive cell with NEP ...")
    relax_structure(prim, fmax=1e-5)
    write(work_dir / "POSCAR", prim)

    os.chdir(work_dir)
    print(f"[{name}] Generating displacements (FC2 + FC3) ...")
    Phono3pyYaml()  # ensure module is loaded
    unitcell, _ = read_crystal_structure("POSCAR", interface_mode="vasp")
    ph3 = Phono3py(unitcell, supercell_matrix=dim, primitive_matrix="auto")

    ph3.generate_displacements()
    ph3.save("phono3py_disp.yaml")
    disp_dataset = ph3.dataset
    supercells = ph3.supercells_with_displacements

    print(f"[{name}] Calculating forces for {len(supercells)} supercells ...")
    forces_data = []
    for it, ph_sc in enumerate(supercells):
        structure_sc = _phonopy_to_ase_atoms(ph_sc)
        structure_sc.calc = build_nep_calculator(nep_path)
        forces = structure_sc.get_forces()
        forces_data.append(forces)
        if it % 100 == 0:
            print(
                f"[{name}] FC3 supercell {it:5d} / {len(supercells)}, "
                f"f_max = {np.max(np.abs(forces)):8.5f} eV/Å",
            )

    forces_data = np.array(forces_data).reshape(-1, 3)
    np.savetxt("FORCES_FC3", forces_data)
    forces = np.loadtxt("FORCES_FC3").reshape(-1, len(ph3.supercell), 3)
    ph3.dataset = disp_dataset
    ph3.forces = forces

    print(f"[{name}] Producing FC2 and FC3 ...")
    ph3.produce_fc2()
    write_fc2_to_hdf5(ph3.fc2)

    ph3.produce_fc3()
    write_fc3_to_hdf5(ph3.fc3)

    print(f"[{name}] Running BTE with mesh = {mesh} ...")
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=range(T_min, T_max + T_step, T_step),
        write_kappa=True,
    )

    os.chdir(work_root)
    kappa_file = work_dir / f"kappa-m{mesh[0]}{mesh[1]}{mesh[2]}.hdf5"
    fobj = h5py.File(kappa_file, "r")

    temperatures = fobj["temperature"][:]
    kappa = fobj["kappa"][:]
    gamma = fobj["gamma"][:]
    freqs = fobj["frequency"][:]
    fobj.close()

    rows = []
    for i, T in enumerate(temperatures):
        rows.append(
            dict(
                structure=name,
                T=T,
                kappa_x=kappa[i, 0],
                kappa_y=kappa[i, 1],
                kappa_z=kappa[i, 2],
            )
        )
    df_kappa = DataFrame(rows)
    csv_path = work_root / f"{name}_kappa_vs_T.csv"
    df_kappa.to_csv(csv_path, index=False)
    print(f"[{name}] kappa(T) data saved to {csv_path}")

    fig, ax = plt.subplots()
    ax.plot(temperatures, kappa[:, 0], "-", label=r"$\\kappa_{xx}$")
    ax.plot(temperatures, kappa[:, 1], "--", label=r"$\\kappa_{yy}$")
    ax.plot(temperatures, kappa[:, 2], "-.", label=r"$\\kappa_{zz}$")

    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim([min(temperatures), temperatures.max()])
    ax.set_ylim([0, np.max(kappa) * 1.05])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Thermal conductivity (W/m/K)")
    ax.set_title(name)

    plt.tight_layout()
    fig_kappa = work_root / f"{name}_kappa_vs_T.png"
    plt.savefig(fig_kappa, dpi=300)
    plt.close()
    print(f"[{name}] κ(T) figure saved to {fig_kappa}")

    fig, ax = plt.subplots()
    base_indices = [9, 29, 99]
    T_indices = [i for i in base_indices if i < len(temperatures)]
    for idx in T_indices:
        Tval = temperatures[idx]
        g = gamma[idx].flatten()
        g = np.where(g > 0.0, g, -1)
        lifetimes = np.where(g > 0.0, 1.0 / (2 * 2 * np.pi * g), np.nan)
        ax.semilogy(
            freqs.flatten(),
            lifetimes,
            "o",
            label=f"T={Tval:.0f} K",
            alpha=0.5,
            markersize=2,
        )

    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Phonon frequency (THz)")
    ax.set_ylabel("Phonon lifetime (ps)")
    ax.set_xlim([0, freqs.max() * 1.02])
    ax.set_title(name)

    plt.tight_layout()
    fig_tau = work_root / f"{name}_lifetimes_vs_frequency.png"
    plt.savefig(fig_tau, dpi=300)
    plt.close()
    print(f"[{name}] lifetime figure saved to {fig_tau}")

    return df_kappa
