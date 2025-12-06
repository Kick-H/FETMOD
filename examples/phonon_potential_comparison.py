"""Compare phonon spectra produced by multiple potentials for silicon.

This example shows how to call :func:`semicon_workflow.phonon_properties.compute_phonon_band_and_dos`
with different calculators, enabling side-by-side comparison of band structures
and densities of states generated with DFT, NEP, Stillinger-Weber, and Tersoff
potentials.

It is intentionally verbose so users can adapt each calculator block to their
own environment (paths to potential files, DFT settings, etc.). Only the
selected calculators that can be built successfully will run; others will be
skipped with a message.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Dict

from ase.build import bulk

from semicon_workflow.calculators import build_nep_calculator
from semicon_workflow.phonon_properties import compute_phonon_band_and_dos

# Update the paths below to match your environment
NEP_PATH = Path("/path/to/nep.txt")
SW_POTENTIAL = Path("/path/to/Si.sw")
TERSOFF_POTENTIAL = Path("/path/to/Si.tersoff")
QE_PSEUDO = Path("/path/to/Si.pbe-n-rrkjus_psl.1.0.0.UPF")


# Calculator builders -------------------------------------------------------
def build_nep() -> object:
    """NEP calculator via calorine (requires NEP_PATH)."""
    if not NEP_PATH.is_file():
        raise FileNotFoundError(f"NEP potential not found: {NEP_PATH}")
    return build_nep_calculator(str(NEP_PATH))


def _build_lammps_calculator(pair_style: str, potential: Path) -> object:
    """Create a LAMMPS calculator using a single silicon potential file.

    This helper mirrors the manual setup you would do when running LAMMPS
    directly: the potential file is passed to ``pair_coeff`` and also copied
    into the working directory via the ``files`` argument. Returning a fully
    constructed calculator (instead of a raw ``parameters`` dict) avoids the
    ``expected str, bytes or os.PathLike object, not dict`` error that occurs
    when :func:`compute_phonon_band_and_dos` tries to attach a calculator.
    """

    if importlib.util.find_spec("ase.calculators.lammpsrun") is None:
        raise RuntimeError("Install ASE's LAMMPS interface to run the LAMMPS examples")
    from ase.calculators.lammpsrun import LAMMPS  # type: ignore

    if not potential.is_file():
        raise FileNotFoundError(f"Potential not found: {potential}")

    parameters = {
        "pair_style": pair_style,
        "pair_coeff": [f"* * {str(potential)} Si"],
        "mass": ["1 28.0855"],
    }
    return LAMMPS(parameters=parameters, files=[str(potential)])


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


# Runner -------------------------------------------------------------------
def run_all(potential_builders: Dict[str, Callable[[], object]]) -> None:
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    out_root = Path("results") / "phonon_comparison"

    for name, builder in potential_builders.items():
        label = f"Si_{name}"
        print(f"\n=== {label}: starting phonon workflow ===")
        try:
            compute_phonon_band_and_dos(
                atoms=si,
                nep_path=None,
                supercell=(2, 2, 2),
                mesh=80,
                name=label,
                outdir=out_root / name,
                calc_builder=builder,
                relax_fmax=5e-4,
            )
            print(f"=== {label}: completed successfully ===")
        except Exception as exc:  # noqa: BLE001
            print(f"=== {label}: skipped ({exc}) ===")


if __name__ == "__main__":
    POTENTIALS: Dict[str, Callable[[], object]] = {
        "DFT_QE": build_qe_dft,
        "NEP": build_nep,
        "SW": build_sw,
        "Tersoff": build_tersoff,
    }
    run_all(POTENTIALS)
