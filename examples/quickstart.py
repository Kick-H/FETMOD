"""Minimal example for running the semicon_workflow package.

This script shows how to:
- Fetch a structure from the Materials Project.
- Run lattice, elastic, phonon, and thermal-conductivity calculations.
- Override the default property list or provide a custom ASE structure.

Prerequisites:
- A Materials Project API key in the environment (PMG_MAPI_KEY or MP_API_KEY).
- A NEP potential file path available to the workflow (update `nep_path`).
- Required runtime dependencies installed (calorine, pymatgen, phonopy, phono3py, etc.).

Note: BTE calculations can be expensive; start with small meshes for testing.
"""

from __future__ import annotations

from ase.build import bulk

from semicon_workflow.workflow import MaterialSpec, run_material_workflow


if __name__ == "__main__":
    # Update this to your NEP potential file
    NEP_PATH = "/path/to/nep.txt"

    # Example 1: Fetch structure from Materials Project by formula
    si_spec = MaterialSpec(
        formula="Si",
        name="Si",
        nep_path=NEP_PATH,
        phonon_supercell=(2, 2, 2),
        phonon_mesh=80,
        bte_dim=(3, 3, 3),
        bte_mesh=(8, 8, 8),
    )

    run_material_workflow(
        mat=si_spec,
        properties=("lattice", "elastic", "phonon"),  # add "thermal_conductivity" when ready
        mp_api_key=None,  # read from environment
        out_root="results",
    )

    # Example 2: Provide your own ASE structure (bypassing MP fetch)
    # Here we build a conventional silicon cell; you can replace this with your own Atoms object
    custom_atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    custom_spec = MaterialSpec(
        formula="Si",  # still used for folder naming
        name="Si_custom",
        nep_path=NEP_PATH,
    )

    # Monkeypatch the workflow fetcher if you want to use a local structure only
    from semicon_workflow import workflow

    def _use_local_structure(*_, **__):
        return workflow.MPStructureSet(
            formula="Si",
            primitive=custom_atoms,
            conventional=custom_atoms,
            orthogonal=custom_atoms,
        )

    workflow.fetch_structures_from_mp = _use_local_structure

    run_material_workflow(
        mat=custom_spec,
        properties=("lattice", "elastic"),
        mp_api_key=None,
        out_root="results_custom",
    )
