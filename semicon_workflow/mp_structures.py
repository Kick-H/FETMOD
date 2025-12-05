"""Utilities for fetching and converting structures from Materials Project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import os

from ase import Atoms
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pymatgen.transformations.advanced_transformations import (
        CubicSupercellTransformation,
    )
except ImportError:  # pragma: no cover - optional dependency pattern
    CubicSupercellTransformation = None

CellType = Literal["primitive", "conventional", "orthogonal"]


@dataclass
class MPStructureSet:
    """Container for the three common cell choices returned by MP queries."""

    formula: str
    primitive: Atoms
    conventional: Atoms
    orthogonal: Optional[Atoms]


def _get_lowest_energy_structure(mpr: MPRester, formula: str):
    """Return the lowest-energy Materials Project structure for ``formula``."""
    docs = mpr.summary_search(
        formula=formula,
        _fields=["material_id", "energy_per_atom", "structure"],
    )
    docs = [d for d in docs if d.get("energy_per_atom") is not None]
    if not docs:
        raise RuntimeError(f"No valid structures found for {formula}")
    best = min(docs, key=lambda d: d["energy_per_atom"])
    return best["structure"], best["material_id"], best["energy_per_atom"]


def _make_orthogonal_cell(struct, max_atoms: int = 400):
    """Generate an orthogonal cell when possible using cubic-like supercell transforms."""
    if CubicSupercellTransformation is None:
        return None

    alpha, beta, gamma = struct.lattice.angles
    if (
        abs(alpha - 90) < 1e-3
        and abs(beta - 90) < 1e-3
        and abs(gamma - 90) < 1e-3
    ):
        return struct

    a, b, c = struct.lattice.abc
    min_length = min(a, b, c)
    max_length = 3.0 * max(a, b, c)

    transform = CubicSupercellTransformation(
        min_atoms=None,
        max_atoms=max_atoms,
        min_length=min_length,
        max_length=max_length,
        force_diagonal=False,
        force_90_degrees=True,
        allow_orthorhombic=True,
    )

    try:
        return transform.apply_transformation(struct)
    except Exception:
        return None


def fetch_structures_from_mp(
    formula: str,
    api_key: Optional[str] = None,
    max_atoms_ortho: int = 400,
) -> MPStructureSet:
    """Fetch primitive/conventional/orthogonal cells from Materials Project.

    Parameters
    ----------
    formula:
        Chemical formula used for the Materials Project query.
    api_key:
        Optional API key. Falls back to ``PMG_MAPI_KEY`` or ``MP_API_KEY``
        environment variables when not supplied.
    max_atoms_ortho:
        Upper bound on atoms in the generated orthogonal supercell.
    """
    if api_key is None:
        api_key = os.environ.get("PMG_MAPI_KEY") or os.environ.get("MP_API_KEY")
    if api_key is None:
        raise RuntimeError("Please set PMG_MAPI_KEY or MP_API_KEY for Materials Project.")

    adaptor = AseAtomsAdaptor()

    with MPRester(api_key) as mpr:
        pmg_struct, mpid, epa = _get_lowest_energy_structure(mpr, formula)
        print(f"[MP] {formula}: use {mpid}, E/atom = {epa:.6f} eV")

        sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-3)
        pmg_prim = sga.get_primitive_standard_structure()
        pmg_conv = sga.get_conventional_standard_structure()
        pmg_ortho = _make_orthogonal_cell(pmg_conv, max_atoms_ortho)

        prim_atoms = adaptor.get_atoms(pmg_prim)
        conv_atoms = adaptor.get_atoms(pmg_conv)
        ortho_atoms = adaptor.get_atoms(pmg_ortho) if pmg_ortho is not None else None

        for atoms in (prim_atoms, conv_atoms, ortho_atoms):
            if atoms is not None:
                atoms.pbc = True

        return MPStructureSet(
            formula=formula,
            primitive=prim_atoms,
            conventional=conv_atoms,
            orthogonal=ortho_atoms,
        )
