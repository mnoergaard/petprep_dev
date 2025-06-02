import nibabel as nb
import numpy as np

from ..pvc import init_pet_pvc_wf


def test_pvc_connections(tmp_path):
    pet_img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    seg_img = nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    seg_file = tmp_path / 'seg.nii.gz'
    pet_img.to_filename(pet_file)
    seg_img.to_filename(seg_file)

    wf = init_pet_pvc_wf()
    wf.base_dir = tmp_path
    wf.inputs.inputnode.pet_t1w = str(pet_file)
    wf.inputs.inputnode.t1w_dseg = str(seg_file)
    wf.inputs.inputnode.petref2anat_xfm = str(seg_file)
    wf.inputs.inputnode.pvc_method = 'STC'
    wf.inputs.inputnode.psf = (1.0, 1.0, 1.0)

    edge = wf._graph.get_edge_data(wf.get_node('make_masks'), wf.get_node('pvc'))
    assert ('out_file', 'mask_file') in edge['connect']
