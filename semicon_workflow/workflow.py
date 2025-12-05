"""High-level orchestration for running NEP-based materials workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

from ase import Atoms

from .elastic_properties import compute_elastic_constants_and_scan
from .mp_structures import MPStructureSet, fetch_structures_from_mp
from .phonon_properties import compute_phonon_band_and_dos
from .relax_lattice import relax_cell_and_scan_a
from .thermal_conductivity_bte import compute_kappa_bte

PropertyType = Literal[
    "lattice",
    "elastic",
    "phonon",
    "thermal_conductivity",
    "thermal_expansion",
]


@dataclass
class MaterialSpec:
    """Input specification for a single material workflow run."""

    formula: str
    name: Optional[str]
    nep_path: str
    phonon_supercell: Sequence[int] = (2, 2, 2)
    phonon_mesh: int = 140
    bte_dim: Sequence[int] = (3, 3, 3)
    bte_mesh: Sequence[int] = (10, 10, 10)


def run_material_workflow(
    mat: MaterialSpec,
    properties: Sequence[PropertyType] = (
        "lattice",
        "elastic",
        "phonon",
        "thermal_conductivity",
    ),
    mp_api_key: Optional[str] = None,
    out_root: str | Path = "results",
):
    """Run a suite of property calculations for a single material."""
    out_root = Path(out_root)
    name = mat.name or mat.formula
    print(f"\n========== Running workflow for {name} ==========")

    mp_structs: MPStructureSet = fetch_structures_from_mp(
        formula=mat.formula,
        api_key=mp_api_key,
    )
    atoms_prim: Atoms = mp_structs.primitive
    atoms_ortho: Atoms = mp_structs.orthogonal or mp_structs.conventional

    if "lattice" in properties:
        run_dir = out_root / name / "lattice"
        relax_cell_and_scan_a(
            atoms=atoms_ortho,
            nep_path=mat.nep_path,
            name=name,
            outdir=run_dir,
        )

    if "elastic" in properties:
        run_dir = out_root / name / "elastic"
        compute_elastic_constants_and_scan(
            atoms=atoms_ortho,
            nep_path=mat.nep_path,
            supercell=mat.phonon_supercell,
            name=name,
            outdir=run_dir,
        )

    if "phonon" in properties:
        run_dir = out_root / name / "phonon"
        compute_phonon_band_and_dos(
            atoms=atoms_ortho,
            nep_path=mat.nep_path,
            supercell=mat.phonon_supercell,
            mesh=mat.phonon_mesh,
            name=name,
            outdir=run_dir,
        )

    if "thermal_conductivity" in properties:
        run_dir = out_root / name / "thermal_conductivity"
        compute_kappa_bte(
            prim=atoms_prim,
            nep_path=mat.nep_path,
            dim=tuple(mat.bte_dim),
            mesh=tuple(mat.bte_mesh),
            name=name,
            work_root=run_dir,
        )

    if "thermal_expansion" in properties:
        print(
            f"[{name}] thermal_expansion: use lattice + phonon modules to generate "
            "E(V) and F_vib(V,T), then feed them into thermal_expansion.py's QHA "
            "helper for α(T)."
        )


if __name__ == "__main__":
    mats = [
        MaterialSpec(formula="Si", name="Si", nep_path="/path/to/nep.txt"),
        MaterialSpec(formula="SiO2", name="SiO2", nep_path="/path/to/nep.txt"),
        MaterialSpec(formula="TiN", name="TiN", nep_path="/path/to/nep.txt"),
    ]

    for m in mats:
        run_material_workflow(
            mat=m,
            properties=("lattice", "elastic", "phonon", "thermal_conductivity"),
            mp_api_key=None,
            out_root="results",
        )
