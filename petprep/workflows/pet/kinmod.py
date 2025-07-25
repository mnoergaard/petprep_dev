# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflow for regional kinetic modeling."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ... import config
from ...interfaces import DerivativesDataSink, FitKineticModel


def init_pet_kinmod_wf(
    *,
    tacs_file: str,
    blood_file: str,
    models: list[str],
    tstar: float | None = None,
    vb_fixed: float | None = None,
    fit_end_time: float | None = None,
    inpshift: float | None = None,
    n_iterations: int | None = None,
    save_figures: bool = False,
    name: str = "pet_kinmod_wf",
) -> pe.Workflow:
    """Fit kinetic models from regional TACs and blood data."""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["tacs_file", "blood_file"]),
        name="inputnode",
    )
    inputnode.inputs.tacs_file = tacs_file
    inputnode.inputs.blood_file = blood_file

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[f"{m}_params" for m in models]),
        name="outputnode",
    )

    for model in models:
        fit = pe.Node(FitKineticModel(model=model), name=f"fit_{model}")
        if tstar is not None and model in {"logan", "ma1"}:
            fit.inputs.t_star = tstar
        if vb_fixed is not None and model in {"1tcm", "2tcm"}:
            fit.inputs.vB_fixed = vb_fixed
        if fit_end_time is not None and model in {"1tcm", "2tcm"}:
            fit.inputs.fit_end_time = fit_end_time
        if inpshift is not None and model == "2tcm":
            fit.inputs.inpshift = inpshift
        if n_iterations is not None:
            fit.inputs.n_iterations = n_iterations
        if save_figures:
            fit.inputs.save_figures = True

        ds_params = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                seg=config.workflow.seg,
                model=model,
                allowed_entities=("seg", "model"),
                suffix="kinpar",
                extension=".tsv",
            ),
            name=f"ds_{model}_kinpar",
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        ds_meta = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                seg=config.workflow.seg,
                model=model,
                allowed_entities=("seg", "model"),
                suffix="kinpar",
                extension=".json",
            ),
            name=f"ds_{model}_kinpar_json",
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        workflow.connect(
            [
                (inputnode, fit, [("tacs_file", "tacs_file"), ("blood_file", "blood_file")]),
                (fit, ds_params, [("params_file", "in_file")]),
                (fit, ds_meta, [("metadata_file", "in_file")]),
                (inputnode, ds_params, [("tacs_file", "source_file")]),
                (inputnode, ds_meta, [("tacs_file", "source_file")]),
                (fit, outputnode, [("params_file", f"{model}_params")]),
            ]
        )

    return workflow


__all__ = ("init_pet_kinmod_wf",)
