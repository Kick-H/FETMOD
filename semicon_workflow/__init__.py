"""Integrated NEP-based materials workflow package.

Top-level imports are guarded so the package can be imported even when optional
heavy dependencies (calorine, pymatgen, phono3py, matplotlib, etc.) are not
installed. Call sites should check for ``None`` where appropriate or import
submodules directly.
"""

# Optional imports with graceful degradation
try:
    from .calculators import build_nep_calculator
except Exception:  # pragma: no cover - optional dependency guard
    build_nep_calculator = None

try:  # pragma: no cover - optional dependency guard
    from .elastic_properties import compute_elastic_constants_and_scan
except Exception:
    compute_elastic_constants_and_scan = None

try:  # pragma: no cover - optional dependency guard
    from .mp_structures import MPStructureSet, fetch_structures_from_mp
except Exception:
    MPStructureSet = None
    fetch_structures_from_mp = None

try:  # pragma: no cover - optional dependency guard
    from .phonon_properties import compute_phonon_band_and_dos
except Exception:
    compute_phonon_band_and_dos = None

try:  # pragma: no cover - optional dependency guard
    from .plotting import set_nature_style
except Exception:
    set_nature_style = None

try:  # pragma: no cover - optional dependency guard
    from .relax_lattice import LatticeScanResult, relax_cell_and_scan_a
except Exception:
    LatticeScanResult = None
    relax_cell_and_scan_a = None

try:  # pragma: no cover - optional dependency guard
    from .thermal_conductivity_bte import compute_kappa_bte
except Exception:
    compute_kappa_bte = None

try:  # pragma: no cover - optional dependency guard
    from .thermal_expansion import ThermalExpansionResult, compute_thermal_expansion_qha
except Exception:
    ThermalExpansionResult = None
    compute_thermal_expansion_qha = None

try:  # pragma: no cover - optional dependency guard
    from .workflow import MaterialSpec, run_material_workflow
except Exception:
    MaterialSpec = None
    run_material_workflow = None

__all__ = [
    "build_nep_calculator",
    "compute_elastic_constants_and_scan",
    "MPStructureSet",
    "fetch_structures_from_mp",
    "compute_phonon_band_and_dos",
    "set_nature_style",
    "LatticeScanResult",
    "relax_cell_and_scan_a",
    "compute_kappa_bte",
    "ThermalExpansionResult",
    "compute_thermal_expansion_qha",
    "MaterialSpec",
    "run_material_workflow",
]
