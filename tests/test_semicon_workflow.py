import sys
from types import SimpleNamespace

import pytest

import pathlib
import sys as _sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import semicon_workflow
from semicon_workflow.thermal_expansion import compute_thermal_expansion_qha


@pytest.fixture(autouse=True)
def reset_rcparams():
    # Ensure rcParams start from defaults for each test
    pytest.importorskip("matplotlib")
    import matplotlib

    with matplotlib.rc_context():
        yield


def test_set_nature_style_updates_rcparams():
    if semicon_workflow.set_nature_style is None:
        pytest.skip("matplotlib not available")

    import matplotlib.pyplot as plt

    semicon_workflow.set_nature_style()
    assert tuple(plt.rcParams["figure.figsize"]) == (3.5, 2.8)
    assert plt.rcParams["font.size"] == 8


def test_compute_thermal_expansion_qha_requires_phonopy(monkeypatch):
    from semicon_workflow import thermal_expansion

    monkeypatch.setattr(thermal_expansion, "_HAS_QHA", False)

    atoms = SimpleNamespace()  # placeholder, not used when _HAS_QHA is False
    with pytest.raises(RuntimeError):
        compute_thermal_expansion_qha(
            atoms=atoms,
            nep_path="nep.txt",
            volumes=[1.0],
            energies=[0.0],
            phonon_free_energies=[[0.0]],
            temperatures=[300.0],
        )


def test_workflow_invokes_modules(monkeypatch):
    pytest.importorskip("ase")
    pytest.importorskip("calorine")
    pytest.importorskip("pymatgen")

    from ase import Atoms
    import numpy as np

    from semicon_workflow.workflow import MaterialSpec, run_material_workflow
    from semicon_workflow.mp_structures import MPStructureSet

    atoms = Atoms("Si2", positions=[[0, 0, 0], [0.25, 0.25, 0.25]], cell=np.eye(3), pbc=True)

    # Collect calls for verification
    called = {"lattice": False, "elastic": False, "phonon": False, "thermal_conductivity": False}

    def fake_fetch_structures_from_mp(*args, **kwargs):
        return MPStructureSet(formula="Si", primitive=atoms, conventional=atoms, orthogonal=atoms)

    def mark(name):
        def _fn(*args, **kwargs):
            called[name] = True
            return None
        return _fn

    monkeypatch.setattr("semicon_workflow.workflow.fetch_structures_from_mp", fake_fetch_structures_from_mp)
    monkeypatch.setattr("semicon_workflow.workflow.relax_cell_and_scan_a", mark("lattice"))
    monkeypatch.setattr("semicon_workflow.workflow.compute_elastic_constants_and_scan", mark("elastic"))
    monkeypatch.setattr("semicon_workflow.workflow.compute_phonon_band_and_dos", mark("phonon"))
    monkeypatch.setattr("semicon_workflow.workflow.compute_kappa_bte", mark("thermal_conductivity"))

    spec = MaterialSpec(formula="Si", name="Si", nep_path="nep.txt")

    run_material_workflow(
        mat=spec,
        properties=("lattice", "elastic", "phonon", "thermal_conductivity"),
        mp_api_key="dummy",
        out_root="/tmp/semicon-test",
    )

    assert all(called.values())
