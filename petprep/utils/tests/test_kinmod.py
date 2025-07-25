import json
from pathlib import Path

import pandas as pd
import pytest

from petprep import data
from petprep.utils.kinmod import (
    load_tacs,
    load_blood,
    save_kinpar_tsv,
    save_kinpar_json,
)


def test_load_tacs():
    tsv = data.load(
        'tests/ds000005/derivatives/petprep/sub-01/pet/sub-01_desc-preproc_seg-gtm_timeseries.tsv'
    )
    df = load_tacs(tsv)
    expected = pd.read_csv(tsv, sep='\t')
    pd.testing.assert_frame_equal(df, expected)


def test_load_blood():
    tsv = data.load(
        'tests/ds000005/derivatives/bloodstream/sub-01/pet/sub-01_inputfunction.tsv'
    )
    df = load_blood(tsv)
    expected = pd.read_csv(tsv, sep='\t')
    pd.testing.assert_frame_equal(df, expected)


def test_save_kinpar_io(tmp_path):
    base = data.load('tests/ds000005/derivatives/petprep/sub-01/pet')
    tsv_file = base / 'sub-01_seg-gtm_model-1tcm_kinpar.tsv'
    json_file = base / 'sub-01_seg-gtm_model-1tcm_kinpar.json'

    df = pd.read_csv(tsv_file, sep='\t')
    json_data = json.loads(json_file.read_text())

    out_tsv = save_kinpar_tsv(df, tmp_path / 'kinpar.tsv')
    out_json = save_kinpar_json(json_data, tmp_path / 'kinpar.json')

    pd.testing.assert_frame_equal(pd.read_csv(out_tsv, sep='\t'), df)
    assert json.loads(Path(out_json).read_text()) == json_data

    # also ensure dict input works for TSV
    out_tsv2 = save_kinpar_tsv(df.iloc[0].to_dict(), tmp_path / 'kinpar_row.tsv')
    df2 = pd.read_csv(out_tsv2, sep='\t')
    pd.testing.assert_frame_equal(df2, df.iloc[[0]])
