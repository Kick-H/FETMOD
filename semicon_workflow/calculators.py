"""Calculator builders used throughout the workflow."""
from calorine.calculators import CPUNEP


def build_nep_calculator(nep_path: str) -> CPUNEP:
    """Return a CPUNEP calculator instance for the given potential file."""
    return CPUNEP(nep_path)
