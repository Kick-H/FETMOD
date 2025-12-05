"""CLI wrapper for the packaged GAAFET generator.

This module now delegates to :func:`semicon_workflow.gaa_structure.cli_generate_gaafet`
so the same logic can be reused programmatically.
"""
from semicon_workflow.gaa_structure import cli_generate_gaafet


if __name__ == "__main__":
    cli_generate_gaafet()
