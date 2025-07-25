# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Utility functions for kinetic modeling data."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def load_tacs(tsv_file: str | Path) -> pd.DataFrame:
    """Load time-activity curves from a TSV file."""
    return pd.read_csv(tsv_file, sep="\t")


def load_blood(tsv_file: str | Path) -> pd.DataFrame:
    """Load blood data from a TSV file."""
    return pd.read_csv(tsv_file, sep="\t")


def save_kinpar_tsv(data: dict | pd.DataFrame, out_file: str | Path) -> Path:
    """Save kinetic parameters to a TSV file."""
    out_file = Path(out_file)
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
    df.to_csv(out_file, sep="\t", index=False)
    return out_file


def save_kinpar_json(data: dict, out_file: str | Path) -> Path:
    """Save kinetic parameters to a JSON file."""
    out_file = Path(out_file)
    out_file.write_text(json.dumps(data, indent=2))
    return out_file
