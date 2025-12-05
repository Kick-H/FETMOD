"""Thermal expansion coefficient interface and QHA skeleton implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from pandas import DataFrame

from .plotting import set_nature_style

try:  # pragma: no cover - optional dependency
    from phonopy.qha import QHA

    _HAS_QHA = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_QHA = False


@dataclass
class ThermalExpansionResult:
    """Container for thermal expansion outputs."""

    temperatures: np.ndarray
    alpha: np.ndarray
    volume_T: np.ndarray
    volume_0: float


def compute_thermal_expansion_qha(
    atoms: Atoms,
    nep_path: str,
    volumes: Sequence[float],
    energies: Sequence[float],
    phonon_free_energies: Sequence[Sequence[float]],
    temperatures: Sequence[float],
    is_volume_expansion: bool = True,
    name: str = "material",
    outdir: str | Path = ".",
) -> ThermalExpansionResult:
    """Quasi-harmonic approximation scaffold for thermal expansion."""
    # ``atoms`` and ``nep_path`` are accepted for symmetry with other modules
    # and future extensions that may automate force-constant generation.
    _ = (atoms, nep_path)
    if not _HAS_QHA:
        raise RuntimeError(
            "phonopy.qha not available. Please install phonopy with QHA support "
            "or compute thermal expansion externally using the generated E(V) and "
            "phonon free energy data."
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_nature_style()

    volumes = np.array(volumes)
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    phonon_free_energies = np.array(phonon_free_energies)

    qha = QHA(
        volumes=volumes,
        electronic_energies=energies,
        temperatures=temperatures,
        free_energies=phonon_free_energies,
    )

    volume_T = qha.volume_temperature()
    volume_0 = volumes[np.argmin(energies)]

    dVdT = np.gradient(volume_T, temperatures)
    alpha_v = dVdT / volume_T
    alpha = alpha_v if is_volume_expansion else alpha_v / 3.0

    df = DataFrame({"T": temperatures, "alpha": alpha, "V_T": volume_T})
    df.to_csv(outdir / f"{name}_thermal_expansion.csv", index=False)

    return ThermalExpansionResult(
        temperatures=temperatures,
        alpha=alpha,
        volume_T=volume_T,
        volume_0=volume_0,
    )
