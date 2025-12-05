# FETMOD

A modular workflow package for NEP-based materials calculations. The
`semicon_workflow` package groups together structure retrieval, relaxation,
elastic, phonon, thermal transport, and thermal expansion interfaces so the
former standalone scripts can be orchestrated from a single entry point.

## Package layout

```
semicon_workflow/
├── calculators.py              # NEP calculator factory
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

Two runnable examples live under `examples/`:

- `examples/quickstart.py` fetches silicon from the Materials Project and runs
  the lattice/elastic/phonon portions of the workflow, and also shows how to
  bypass MP fetching by supplying a local ASE structure.
- `examples/gaa_structure_generator.py` now delegates to the packaged
  `semicon_workflow.gaa_structure` utilities so you can generate GAAFET/FinFET
  structures from the library or via the CLI.

Run them with your NEP path and MP API key set in the environment:

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

Ke Xu, Gang Wang, Ting Liang, Yang Xiao, Dongliang Ding, Haichang Guo, Xiang Gao, Lei Tong, Xi Wan, Gang Zhang and Jianbin Xu， Device-Scale Atomistic Simulations of Heat Transport in Advanced Field-Effect Transistors, 2025, (https://arxiv.org/abs/2511.18915)[https://arxiv.org/abs/2511.18915]
