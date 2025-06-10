from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ...interfaces import GTMSeg


def init_pet_gtmseg_wf(name: str = 'pet_gtmseg_wf') -> pe.Workflow:
    """Generate GTM segmentation from FreeSurfer reconstructions."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subjects_dir', 'subject_id']),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=['gtmseg']), name='outputnode')

    gtmseg = pe.Node(GTMSeg(), name='gtmseg')

    workflow.connect(
        [
            (inputnode, gtmseg, [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')]),
            (gtmseg, outputnode, [('out_file', 'gtmseg')]),
        ]
    )

    return workflow
