from ..seg import init_pet_gtmseg_wf


def test_gtmseg_connections():
    wf = init_pet_gtmseg_wf()

    edge_in = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('gtmseg'))
    assert ('subjects_dir', 'subjects_dir') in edge_in['connect']
    assert ('subject_id', 'subject_id') in edge_in['connect']

    edge_out = wf._graph.get_edge_data(wf.get_node('gtmseg'), wf.get_node('outputnode'))
    assert ('out_file', 'gtmseg') in edge_out['connect']
