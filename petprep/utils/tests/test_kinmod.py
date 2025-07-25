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

DERIV_DIR = data.load('tests/ds000005/derivatives').absolute()


def test_load_tacs_and_blood():
    tacs_file = DERIV_DIR / 'petprep' / 'sub-01' / 'pet' / 'sub-01_desc-preproc_seg-gtm_timeseries.tsv'
    blood_file = DERIV_DIR / 'bloodstream' / 'sub-01' / 'pet' / 'sub-01_inputfunction.tsv'

    tacs = load_tacs(tacs_file)
    blood = load_blood(blood_file)

    assert tacs.shape == (27, 111)
    assert list(tacs.columns[:2]) == ['FrameTimesStart', 'FrameTimesEnd']
    assert blood.shape == (6000, 5)
    assert list(blood.columns) == [
        'time',
        'whole_blood_radioactivity',
        'plasma_radioactivity',
        'metabolite_parent_fraction',
        'AIF',
    ]


def test_save_kinpar_roundtrip(tmp_path):
    tsv_file = DERIV_DIR / 'petprep' / 'sub-01' / 'pet' / 'sub-01_seg-gtm_model-1tcm_kinpar.tsv'
    json_file = DERIV_DIR / 'petprep' / 'sub-01' / 'pet' / 'sub-01_seg-gtm_model-1tcm_kinpar.json'

    df = pd.read_csv(tsv_file, sep='\t')
    out_tsv = save_kinpar_tsv(df, tmp_path / 'kinpar.tsv')
    round_df = pd.read_csv(out_tsv, sep='\t')
    pd.testing.assert_frame_equal(df, round_df)

    info = json.loads(json_file.read_text())
    out_json = save_kinpar_json(info, tmp_path / 'kinpar.json')
    round_info = json.loads(out_json.read_text())
    assert info == round_info
