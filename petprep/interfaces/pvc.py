from __future__ import annotations

import os

import nibabel as nb
import numpy as np
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec
from nipype.interfaces.petpvc import PETPVC as _PETPVC
from nipype.utils.filemanip import fname_presuffix


class PVCMake4DInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='Segmentation image')
    out_file = File(desc='4D mask file')


class PVCMake4DOutputSpec(TraitedSpec):
    out_file = File(desc='4D mask file')
    index_file = File(desc='Text file with label values')


class PVCMake4D(SimpleInterface):
    """Create a 4D mask file from a segmentation."""

    input_spec = PVCMake4DInputSpec
    output_spec = PVCMake4DOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.asanyarray(img.dataobj)
        labels = [int(label) for label in np.unique(data) if label != 0]

        mask = np.stack(
            [(data == label).astype(np.uint8) for label in labels], axis=-1
        )
        out_file = self.inputs.out_file
        if not out_file:
            out_file = fname_presuffix(self.inputs.in_file, suffix='_masks4d', newpath=runtime.cwd)
        nb.Nifti1Image(mask, img.affine, img.header).to_filename(out_file)

        index_file = os.path.join(runtime.cwd, 'mask_index.txt')
        with open(index_file, 'w') as f:
            for label in labels:
                f.write(f'{label}\n')

        self._results['out_file'] = out_file
        self._results['index_file'] = index_file
        return runtime


class PETPVC(_PETPVC):
    """Thin wrapper around Nipype's PETPVC interface."""
