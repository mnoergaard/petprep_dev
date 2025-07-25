from pathlib import Path

from petprep import data, config
from petprep.workflows.pet.kinmod import init_pet_kinmod_wf
from ...tests import mock_config

DERIV_DIR = data.load('tests/ds000005/derivatives').absolute()


def test_kinmod_connections():
    tacs = DERIV_DIR / 'petprep' / 'sub-01' / 'pet' / 'sub-01_desc-preproc_seg-gtm_timeseries.tsv'
    blood = DERIV_DIR / 'bloodstream' / 'sub-01' / 'pet' / 'sub-01_inputfunction.tsv'

    with mock_config(bids_dir=data.load('tests/ds000005').absolute()):
        wf = init_pet_kinmod_wf(
            tacs_file=str(tacs),
            blood_file=str(blood),
            models=['logan', 'ma1'],
        )

    assert 'fit_logan' in wf.list_node_names()
    assert 'fit_ma1' in wf.list_node_names()
    assert 'ds_logan_kinpar' in wf.list_node_names()
    assert 'ds_ma1_kinpar' in wf.list_node_names()

    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('fit_logan'))
    assert ('tacs_file', 'tacs_file') in edge['connect']
    assert ('blood_file', 'blood_file') in edge['connect']

    edge_ds = wf._graph.get_edge_data(wf.get_node('fit_logan'), wf.get_node('ds_logan_kinpar'))
    assert ('params_file', 'in_file') in edge_ds['connect']
    ds_node = wf.get_node('ds_logan_kinpar')
    assert ds_node.inputs.model == 'logan'
    assert ds_node.inputs.seg == config.workflow.seg
