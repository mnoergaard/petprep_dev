from pathlib import Path

import pandas as pd
from nipype.pipeline import engine as pe

from petprep.interfaces.kinmodels import FitKineticModel


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
