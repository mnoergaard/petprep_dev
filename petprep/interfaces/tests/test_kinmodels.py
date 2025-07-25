from pathlib import Path

import pandas as pd
from nipype.pipeline import engine as pe

from petprep.interfaces.kinmodels import (
    FitKineticModel,
    MA1Model,
    OneTCMModel,
    TwoTCMModel,
)


def test_fitkineticmodel_logan(tmp_path):
    root = Path(__file__).resolve().parents[2]
    tacs = (
        root
        / 'data/tests/ds000005/derivatives/petprep/sub-01/pet'
        / 'sub-01_desc-preproc_seg-gtm_timeseries.tsv'
    )
    blood = (
        root
        / 'data/tests/ds000005/derivatives/bloodstream/sub-01/pet'
        / 'sub-01_inputfunction.tsv'
    )
    node = pe.Node(
        FitKineticModel(tacs_file=str(tacs), blood_file=str(blood), model='logan', t_star=10.0),
        name='fit',
        base_dir=tmp_path,
    )
    res = node.run()
    out = pd.read_csv(res.outputs.params_file, sep='\t')
    assert 'VT' in out.columns


def _run_model(tmp_path, model_name, params, **extra):
    root = Path(__file__).resolve().parents[2]
    tacs = (
        root
        / 'data/tests/ds000005/derivatives/petprep/sub-01/pet'
        / 'sub-01_desc-preproc_seg-gtm_timeseries.tsv'
    )
    blood = (
        root
        / 'data/tests/ds000005/derivatives/bloodstream/sub-01/pet'
        / 'sub-01_inputfunction.tsv'
    )
    kw = {'model': model_name}
    if model_name in {'logan', 'ma1'}:
        kw['t_star'] = 10.0
    kw.update(extra)
    node = pe.Node(
        FitKineticModel(tacs_file=str(tacs), blood_file=str(blood), **kw),
        name=f'fit_{model_name}',
        base_dir=tmp_path,
    )
    res = node.run()
    out = pd.read_csv(res.outputs.params_file, sep='\t')
    for col in params:
        assert col in out.columns


def test_fitkineticmodel_ma1(tmp_path):
    _run_model(tmp_path, 'ma1', MA1Model.parameters)


def test_fitkineticmodel_1tcm(tmp_path):
    _run_model(tmp_path, '1tcm', OneTCMModel.parameters, n_iterations=1)


def test_fitkineticmodel_2tcm(tmp_path):
    _run_model(tmp_path, '2tcm', TwoTCMModel.parameters, n_iterations=1)
