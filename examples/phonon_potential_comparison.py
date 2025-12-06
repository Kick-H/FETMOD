"""Compare phonon spectra produced by multiple potentials for silicon.

This example shows how to call
:func:`semicon_workflow.phonon_properties.compute_phonon_band_and_dos` with
different calculators, enabling side-by-side comparison of band structures and
density-of-states (DOS) curves generated with DFT, NEP, Stillinger–Weber, and
Tersoff potentials.

After running the phonon workflow for each calculator, the script overlays the
band structures on a single figure using the exported high-symmetry path
positions/labels, and overlays the DOS curves in a separate plot. This makes it
easy to visually compare how different potentials perform for the same
structure.

It is intentionally verbose so users can adapt each calculator block to their
own environment (paths to potential files, DFT settings, etc.). Only the
selected calculators that can be built successfully will run; others will be
skipped with a message.
"""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd
from ase.build import bulk

from semicon_workflow.calculators import build_nep_calculator
from semicon_workflow.phonon_properties import compute_phonon_band_and_dos

# Update the paths below to match your environment
NEP_PATH = Path("/path/to/nep.txt")
SW_POTENTIAL = Path("/path/to/Si.sw")
TERSOFF_POTENTIAL = Path("/path/to/Si.tersoff")
QE_PSEUDO = Path("/path/to/Si.pbe-n-rrkjus_psl.1.0.0.UPF")


# ----------------------------------------------------------------------
# Calculator builders
# ----------------------------------------------------------------------
def build_nep() -> object:
    """NEP calculator via calorine (requires NEP_PATH)."""
    if not NEP_PATH.is_file():
        raise FileNotFoundError(f"NEP potential not found: {NEP_PATH}")
    return build_nep_calculator(str(NEP_PATH))


def _build_lammps_calculator(pair_style: str, potential: Path) -> object:
    if importlib.util.find_spec("ase.calculators.lammpsrun") is None:
        raise RuntimeError("Install ASE's LAMMPS interface to run the LAMMPS examples")

    from ase.calculators.lammpsrun import LAMMPS  # type: ignore

    if not potential.is_file():
        raise FileNotFoundError(f"Potential not found: {potential}")

    element = potential.stem.split(".")[0] or "Si"

    files = [str(potential)]
    parameters = {
        "pair_style": pair_style,                      # "sw" or "tersoff"
        "pair_coeff": [f"* * {str(potential)} {element}"],
        # Default：units="metal", atom_style="atomic"
    }

    return LAMMPS(files=files, **parameters)


def build_sw() -> object:
    """LAMMPS Stillinger–Weber potential for silicon."""
    return _build_lammps_calculator("sw", SW_POTENTIAL)


def build_tersoff() -> object:
    """LAMMPS Tersoff potential for silicon."""
    return _build_lammps_calculator("tersoff", TERSOFF_POTENTIAL)


def build_qe_dft() -> object:
    """Quantum ESPRESSO calculator for silicon DFT reference."""
    if importlib.util.find_spec("ase.calculators.espresso") is None:
        raise RuntimeError("Install Quantum ESPRESSO + ASE espresso interface for DFT")
    from ase.calculators.espresso import Espresso  # type: ignore

    if not QE_PSEUDO.is_file():
        raise FileNotFoundError("Set QE_PSEUDO to a valid UPF pseudopotential")

    pseudopotentials = {"Si": QE_PSEUDO.name}
    input_data = {
        "system": {"ecutwfc": 40, "occupations": "smearing", "smearing": "mv"},
        "electrons": {"conv_thr": 1e-9, "diagonalization": "cg"},
    }

    return Espresso(
        pseudopotentials=pseudopotentials,
        pseudopotential_path=str(QE_PSEUDO.parent),
        input_data=input_data,
        kpts=(4, 4, 4),
        calculation="scf",
    )


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_all(potential_builders: Dict[str, Callable[[], object]]) -> None:
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    out_root = Path("results") / "phonon_comparison"
    comparison_dir = out_root / "comparison"

    runs = []

    for name, builder in potential_builders.items():
        label = f"Si_{name}"
        print(f"\n=== {label}: starting phonon workflow ===")
        try:
            df_band, df_dos = compute_phonon_band_and_dos(
                atoms=si,
                nep_path=None,
                supercell=(2, 2, 2),
                mesh=80,
                name=label,
                outdir=out_root / name,
                calc_builder=builder,
                relax_fmax=5e-4,
            )
            runs.append({"label": label, "run_dir": out_root / name, "band": df_band, "dos": df_dos})
            print(f"=== {label}: completed successfully ===")
        except Exception as exc:  # noqa: BLE001
            print(f"=== {label}: skipped ({exc}) ===")

    if not runs:
        print("No successful runs; cannot build comparison plots.")
        return

    comparison_dir.mkdir(parents=True, exist_ok=True)
    path_df = pd.read_csv(runs[0]["run_dir"] / f"{runs[0]['label']}_phonon_band_path.csv")

    band_results = {run["label"]: run["band"] for run in runs}
    dos_results = {run["label"]: run["dos"] for run in runs}

    _plot_band_overlay(band_results, path_df, comparison_dir / "phonon_band_comparison.png")
    _plot_dos_overlay(dos_results, comparison_dir / "phonon_dos_comparison.png")


def _plot_band_overlay(
    band_results: Dict[str, pd.DataFrame], path_df: pd.DataFrame, outpath: Path
) -> None:
    """Plot multiple phonon band structures on one figure."""

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    fig, ax = plt.subplots()

    for label, df_band in band_results.items():
        color = next(colors)
        for i, col in enumerate(df_band.columns):
            ax.plot(
                df_band.index,
                df_band[col],
                color=color,
                linewidth=0.8,
                label=label if i == 0 else None,
            )

    ax.set_xlim(path_df.positions.min(), path_df.positions.max())
    ax.set_ylabel("Frequency (THz)")
    ax.set_xticks(path_df.positions)
    ax.set_xticklabels(path_df.labels)
    for xp in path_df.positions:
        ax.axvline(xp, color="0.85", linewidth=0.6)

    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Phonon band comparison")
    fig.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved band comparison plot to {outpath}")


def _plot_dos_overlay(dos_results: Dict[str, pd.DataFrame], outpath: Path) -> None:
    """Plot multiple phonon DOS curves on one figure."""

    fig, ax = plt.subplots()
    for label, df_dos in dos_results.items():
        ax.plot(
            df_dos["frequency_THz"],
            df_dos["total_DOS"],
            linewidth=1.0,
            label=label,
        )

    first_df = next(iter(dos_results.values()))
    factor_series = first_df["frequency_cm-1"] / first_df["frequency_THz"]
    factor_clean = factor_series.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
    factor = float(factor_clean.iloc[0]) if not factor_clean.empty else 33.35641

    def thz_to_cm_scalar(x: float) -> float:
        return x * factor

    def cm_to_thz_scalar(x: float) -> float:
        return x / factor

    secax = ax.secondary_xaxis("top", functions=(thz_to_cm_scalar, cm_to_thz_scalar))
    secax.set_xlabel(r"Frequency (cm$^{-1}$)")

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Density of states")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Phonon DOS comparison")
    fig.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved DOS comparison plot to {outpath}")


if __name__ == "__main__":
    POTENTIALS: Dict[str, Callable[[], object]] = {
        #"DFT_QE": build_qe_dft,
        "NEP": build_nep,
        "SW": build_sw,
        "Tersoff": build_tersoff,
    }
    run_all(POTENTIALS)
