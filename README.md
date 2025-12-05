# FETMOD

A modular workflow package for molecular dynamics (MD)-based materials calculations.
The `semicon_workflow` package groups together structure retrieval, relaxation,
elastic, phonon, thermal transport, and thermal expansion interfaces so the
former standalone scripts can be orchestrated from a single entry point.

## Package layout

```
semicon_workflow/
├── calculators.py              # ASE calculator factory
├── elastic_properties.py       # Elastic constants and pressure scan
├── mp_structures.py            # Materials Project fetch + ASE conversion
├── phonon_properties.py        # Phonon band structure and DOS
├── plotting.py                 # Shared Nature-style plotting settings
├── relax_lattice.py            # Cell relaxation and lattice scans
├── thermal_conductivity_bte.py # phono3py BTE workflow
├── thermal_expansion.py        # QHA interface placeholder for α(T)
├── workflow.py                 # High-level orchestration by material
└── __init__.py
```

## Quick start

### Installation

Install the package directly from the repository root:

```bash
pip install .
```

To keep an editable checkout for development work, use:

```bash
pip install -e .[tests]
```

Heavy dependencies are grouped as extras so you can opt into the pieces you
need:

- `pip install .[calculators]` adds the Calorine MD calculators required for
  lattice, elastic, phonon, and thermal-conductivity workflows.
- `pip install .[materials]` pulls in `pymatgen` for Materials Project
  structure fetching and symmetry analysis.
- `pip install .[phonons]` installs the phonopy/phono3py toolchain and HDF5
  helpers needed for the phonon and thermal conductivity interfaces.
- `pip install .[full]` installs all of the above extras.

```python
from semicon_workflow.workflow import MaterialSpec, run_material_workflow

spec = MaterialSpec(formula="Si", name="Si", nep_path="/path/to/nep.txt")
run_material_workflow(
    mat=spec,
    properties=("lattice", "elastic", "phonon", "thermal_conductivity"),
    mp_api_key="<YOUR_MP_KEY>",
    out_root="results",
)
```

The thermal expansion module currently exposes a QHA scaffold so you can feed
precomputed E(V) and phonon free-energy data to obtain α(T). See inline
docstrings for details.

## Examples

Three runnable examples live under `examples/`:

- `examples/quickstart.py` fetches silicon from the Materials Project and runs
  the lattice/elastic/phonon portions of the workflow, and also shows how to
  bypass MP fetching by supplying a local ASE structure.
- `examples/phonon_potential_comparison.py` compares phonon band structures and
  DOS curves for silicon generated with different potentials (DFT/NEP/SW/
  Tersoff). Update the potential and pseudopotential paths before running to
  reproduce the comparison.
- `examples/gaa_structure_generator.py` now delegates to the packaged
  `semicon_workflow.gaa_structure` utilities so you can generate GAAFET/FinFET
  structures from the library or via the CLI.

Run them with your potential file path and MP API key set in the environment:

```bash
python examples/quickstart.py
```

## Tests

A lightweight pytest suite validates the plotting defaults, the QHA guardrails,
and the workflow call graph. Run it with:

```bash
pytest
```

## Cites

Ke Xu, Gang Wang, Ting Liang, Yang Xiao, Dongliang Ding, Haichang Guo, Xiang Gao, Lei Tong, Xi Wan, Gang Zhang and Jianbin Xu， Device-Scale Atomistic Simulations of Heat Transport in Advanced Field-Effect Transistors, 2025, https://arxiv.org/abs/2511.18915
