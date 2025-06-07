"""PET segmentation workflow."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.interfaces.freesurfer import GTMSeg, SegStats

try:
    from nipype.interfaces.freesurfer import (
        SegmentBS,
        SegmentHA_T1,
        SegmentThalamicNuclei,
        MRISclimbicSeg,
        SegmentWM,
    )
except Exception:  # pragma: no cover - fallback definitions
    from nipype.interfaces.freesurfer.base import FSCommand
    from nipype.interfaces.base import (
        TraitedSpec,
        File,
        Directory,
        traits,
        CommandLineInputSpec,
    )

    class _SegInputSpec(CommandLineInputSpec):
        subjects_dir = Directory(exists=True, argstr="--sd %s")
        subject_id = traits.Str(argstr="--s %s", mandatory=True)
        out_file = File(argstr="--o %s")

    class _SegOutputSpec(TraitedSpec):
        out_file = File()

    class SegmentBS(FSCommand):
        _cmd = "segmentBS.sh"
        input_spec = _SegInputSpec
        output_spec = _SegOutputSpec

    class SegmentHA_T1(FSCommand):
        _cmd = "segmentHA_T1.sh"
        input_spec = _SegInputSpec
        output_spec = _SegOutputSpec

    class SegmentThalamicNuclei(FSCommand):
        _cmd = "SegmentThalamicNuclei"
        input_spec = _SegInputSpec
        output_spec = _SegOutputSpec

    class MRISclimbicSeg(FSCommand):
        _cmd = "mri_sclimbic_seg"
        input_spec = _SegInputSpec
        output_spec = _SegOutputSpec

    class SegmentWM(FSCommand):
        _cmd = "mri_segment"
        input_spec = _SegInputSpec
        output_spec = _SegOutputSpec

from ... import config
from ...interfaces import DerivativesDataSink


_DEF_TYPES = [
    "gtm",
    "brainstem",
    "thalamicNuclei",
    "hippocampusAmygdala",
    "wm",
    "raphe",
    "limbic",
]


def _seg_node(seg_type: str):
    if seg_type == "gtm":
        return GTMSeg(), "gtmseg"
    if seg_type == "brainstem":
        return SegmentBS(), "brainstem"
    if seg_type == "thalamicNuclei":
        return SegmentThalamicNuclei(), "thalamicnuclei"
    if seg_type == "hippocampusAmygdala":
        return SegmentHA_T1(), "ha_t1"
    if seg_type == "wm":
        return SegmentWM(), "wm"
    if seg_type == "raphe":
        return MRISclimbicSeg(), "raphe"
    if seg_type == "limbic":
        return MRISclimbicSeg(), "limbic"
    raise ValueError(f"Unknown segmentation type: {seg_type}")


def init_pet_segmentation_wf(
    segmentation_types: list[str] | None = None,
    *,
    name: str = "pet_segmentation_wf",
) -> pe.Workflow:
    """Run FreeSurfer-based segmentation tools."""

    segmentation_types = segmentation_types or []
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["subjects_dir", "subject_id"]), name="inputnode"
    )

    for seg_type in segmentation_types:
        seg_iface, label = _seg_node(seg_type)
        seg = pe.Node(seg_iface, name=f"seg_{label}")
        ds_seg = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc=label,
                suffix="dseg",
            ),
            name=f"ds_{label}",
            run_without_submitting=True,
        )
        stats = pe.Node(SegStats(), name=f"{label}_stats")
        ds_stats = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc=label,
                suffix="stats",
            ),
            name=f"ds_{label}_stats",
            run_without_submitting=True,
        )

        workflow.connect(
            [
                (inputnode, seg, [("subjects_dir", "subjects_dir"), ("subject_id", "subject_id")]),
                (seg, ds_seg, [("out_file", "in_file")]),
                (seg, stats, [("out_file", "segmentation_file")]),
                (stats, ds_stats, [("summary_file", "in_file")]),
            ]
        )

    return workflow
