"""Tracing engine: intervention ops, the trace schema, and derived fields."""

from .interventions import (
    AblateHead,
    DisableCausalMask,
    DisablePositional,
    Intervention,
    InterventionPlan,
    RemoveResidual,
    ScaleSite,
    ZeroMLP,
    compile_interventions,
    parse_interventions,
)
from .summaries import Derived, compute_derived, head_label
from .trace import (
    Output,
    TensorEnvelope,
    Token,
    TopToken,
    Trace,
    TraceRecorder,
    build_trace,
)

__all__ = [
    "AblateHead",
    "Derived",
    "DisableCausalMask",
    "DisablePositional",
    "Intervention",
    "InterventionPlan",
    "Output",
    "RemoveResidual",
    "ScaleSite",
    "TensorEnvelope",
    "Token",
    "TopToken",
    "Trace",
    "TraceRecorder",
    "ZeroMLP",
    "build_trace",
    "compile_interventions",
    "compute_derived",
    "head_label",
    "parse_interventions",
]
