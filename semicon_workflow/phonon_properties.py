"""Phonon band structure and density-of-states calculations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np
from ase import Atoms
from calorine.tools import get_force_constants, relax_structure
from matplotlib import pyplot as plt
from pandas import DataFrame
from phonopy.units import THzToCm
from seekpath import get_explicit_k_path

from .calculators import build_nep_calculator
from .plotting import set_nature_style


def compute_phonon_band_and_dos(
    atoms: Atoms,
    nep_path: str | None = None,
    supercell: Sequence[int] = (2, 2, 2),
    mesh: int = 140,
    dos_range: Tuple[float, float] = (-0.1, 15.0),
    dos_pitch: float = 0.02,
    name: str = "material",
    outdir: str | Path = ".",
    calc_builder: Callable[[], object] | None = None,
    relax_fmax: float = 1.0e-4,
) -> Tuple[DataFrame, DataFrame]:
    """Compute phonon dispersion and total density of states.

    Parameters
    ----------
    atoms
        Structure to evaluate. A copy is made internally.
    nep_path
        Path to a NEP potential file. Required unless ``calc_builder`` is
        provided. Retained for backward compatibility with existing workflow
        calls.
    supercell, mesh, dos_range, dos_pitch, name, outdir
        Simulation parameters forwarded to ``calorine``/``phonopy``.
    calc_builder
        Optional factory returning an ASE-compatible calculator. Use this to
        supply alternative potentials (DFT, SW, Tersoff, etc.).
    relax_fmax
        Maximum force threshold used during structure relaxation.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_nature_style()

    atoms = atoms.copy()
    if calc_builder is None:
        if nep_path is None:
            raise ValueError("Either calc_builder or nep_path must be provided")
        calc_builder = lambda: build_nep_calculator(nep_path)

    atoms.calc = calc_builder()

    print(f"[{name}] Relaxing structure for phonons ...")
    relax_structure(atoms, fmax=relax_fmax)
    print(f"[{name}] Relaxation finished.")

    print(f"[{name}] Computing force constants ...")
    phonon = get_force_constants(atoms, atoms.calc, supercell)

    structure_tuple = (
        atoms.cell,
        atoms.get_scaled_positions(),
        atoms.numbers,
    )
    path = get_explicit_k_path(structure_tuple)

    print(f"[{name}] Running phonon band structure ...")
    phonon.run_band_structure([path["explicit_kpoints_rel"]])
    band = phonon.get_band_structure_dict()
    freqs = band["frequencies"][0]

    df_band = DataFrame(freqs)
    df_band.index = path["explicit_kpoints_linearcoord"]
    df_band.index.name = "k_linear"

    band_csv = outdir / f"{name}_phonon_band.csv"
    df_band.to_csv(band_csv)
    print(f"[{name}] band data saved to {band_csv}")

    fig, ax = plt.subplots()
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    for col in df_band.columns:
        ax.plot(
            df_band.index,
            df_band[col],
            linestyle="-",
            linewidth=0.8,
            marker=None,
            color=color,
        )
    ax.set_xlim(df_band.index.min(), df_band.index.max())
    ax.set_ylabel("Frequency (THz)")

    ax2 = ax.twinx()
    ax2.set_ylabel(r"Frequency (cm$^{-1}$)")
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(THzToCm * np.array([ymin, ymax]))

    labels = path["explicit_kpoints_labels"]
    labels = ["$\\Gamma$" if m == "GAMMA" else m for m in labels]
    labels = [m.replace("_", "$_") + "$" if "_" in m else m for m in labels]
    df_path = DataFrame(dict(labels=labels, positions=path["explicit_kpoints_linearcoord"]))
    df_path.drop(df_path.index[df_path.labels == ""], axis=0, inplace=True)

    ax.set_xticks(df_path.positions)
    ax.set_xticklabels(df_path.labels)
    for xp in df_path.positions:
        ax.axvline(xp, color="0.8", linewidth=0.6)

    ax.set_title(f"{name} phonon band structure")
    plt.tight_layout()
    fig_band = outdir / f"{name}_phonon_band.png"
    plt.savefig(fig_band, dpi=300)
    plt.close()
    print(f"[{name}] band figure saved to {fig_band}")

    dos_min, dos_max = dos_range
    print(f"[{name}] Running phonon DOS ...")
    phonon.run_mesh(mesh)
    phonon.run_total_dos(freq_min=dos_min, freq_max=dos_max, freq_pitch=dos_pitch)
    dos = phonon.get_total_dos_dict()
    freq_points = np.array(dos["frequency_points"])
    total_dos = np.array(dos["total_dos"])

    df_dos = DataFrame(
        {
            "frequency_THz": freq_points,
            "total_DOS": total_dos,
            "frequency_cm-1": freq_points * THzToCm,
        }
    )
    dos_csv = outdir / f"{name}_phonon_dos.csv"
    df_dos.to_csv(dos_csv, index=False)
    print(f"[{name}] DOS data saved to {dos_csv}")

    fig, ax = plt.subplots()
    ax.plot(freq_points, total_dos, linestyle="-", linewidth=1.0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Density of states")

    def thz_to_cm(x):
        return x * THzToCm

    def cm_to_thz(x):
        return x / THzToCm

    secax = ax.secondary_xaxis("top", functions=(thz_to_cm, cm_to_thz))
    secax.set_xlabel(r"Frequency (cm$^{-1}$)")

    ax.set_title(f"{name} phonon DOS")
    plt.tight_layout()
    fig_dos = outdir / f"{name}_phonon_dos.png"
    plt.savefig(fig_dos, dpi=300)
    plt.close()
    print(f"[{name}] DOS figure saved to {fig_dos}")

    return df_band, df_dos
