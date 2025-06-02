from __future__ import annotations

import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.nibabel import MergeSeries, SplitSeries

from ...interfaces import PETPVC, PVCMake4D


def _split_psf(psf):
    if isinstance(psf, (list | tuple)) and len(psf) == 3:
        return psf
    return (psf, psf, psf)


def _compute_tac(pvc_file, dseg):
    import os

    import nibabel as nb

    img = nb.load(pvc_file)
    data = img.get_fdata()
    seg = nb.load(dseg).get_fdata()
    labels = [int(label) for label in np.unique(seg) if label != 0]
    tac = np.zeros((data.shape[-1], len(labels)))
    for i, lab in enumerate(labels):
        mask = seg == lab
        tac[:, i] = data[mask].reshape(-1, data.shape[-1]).mean(axis=0)
    out_file = os.path.abspath('tac.tsv')
    header = '\t'.join(f'label-{label}' for label in labels)
    np.savetxt(out_file, tac, delimiter='\t', header=header, comments='')
    return out_file


def init_pet_pvc_wf(name: str = 'pet_pvc_wf') -> pe.Workflow:
    """Apply partial volume correction to a PET time series."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['pet_t1w', 't1w_dseg', 'petref2anat_xfm', 'pvc_method', 'psf']
        ),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=['pet_pvc', 'tac_file']), name='outputnode')

    make_masks = pe.Node(PVCMake4D(), name='make_masks')
    split = pe.Node(SplitSeries(), name='split')

    psf = pe.Node(niu.Function(function=_split_psf, output_names=['x', 'y', 'z']), name='psf')

    pvc = pe.MapNode(PETPVC(), iterfield=['in_file'], name='pvc')

    merge = pe.Node(MergeSeries(), name='merge')

    tac = pe.Node(niu.Function(function=_compute_tac, output_names=['tac_file']), name='tac')

    workflow.connect(
        [
            (inputnode, make_masks, [('t1w_dseg', 'in_file')]),
            (inputnode, split, [('pet_t1w', 'in_file')]),
            (inputnode, pvc, [('pvc_method', 'pvc')]),
            (inputnode, psf, [('psf', 'psf')]),
            (psf, pvc, [('x', 'fwhm_x'), ('y', 'fwhm_y'), ('z', 'fwhm_z')]),
            (split, pvc, [('out_files', 'in_file')]),
            (make_masks, pvc, [('out_file', 'mask_file')]),
            (pvc, merge, [('out_file', 'in_files')]),
            (merge, tac, [('out_file', 'pvc_file')]),
            (inputnode, tac, [('t1w_dseg', 'dseg')]),
            (merge, outputnode, [('out_file', 'pet_pvc')]),
            (tac, outputnode, [('tac_file', 'tac_file')]),
        ]
    )

    return workflow
