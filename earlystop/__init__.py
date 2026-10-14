"""Forward from zelt for backward compatibility."""

import zelt
from zelt import (
    BatchResult,
    CompareResult,
    EarlyStop,
    EarlyStopResult,
    EvaluationResult,
    StopDecider,
    StreamChunk,
    __version__,
    accelerate,
    ask,
    batch,
    calibrate,
    compare,
    evaluate,
    extract_answer,
    featurize,
    fit_and_report,
    from_ollama,
    from_vllm,
    is_correct,
    list_presets,
    make_stop_fn,
    risk_report,
    same_answer,
    stream,
    token_budget,
)
from zelt.core import EarlyStopper

# Aliases
Zelt = EarlyStop
ZeltResult = EarlyStopResult
reduce_compute_thinking = ask

__all__ = [
    "EarlyStop",
    "EarlyStopResult",
    "StreamChunk",
    "BatchResult",
    "CompareResult",
    "EvaluationResult",
    "EarlyStopper",
    "StopDecider",
    "make_stop_fn",
    "featurize",
    "fit_and_report",
    "extract_answer",
    "is_correct",
    "same_answer",
    "ask",
    "stream",
    "batch",
    "compare",
    "evaluate",
    "calibrate",
    "accelerate",
    "list_presets",
    "from_ollama",
    "from_vllm",
    "token_budget",
    "risk_report",
    "Zelt",
    "ZeltResult",
    "reduce_compute_thinking",
    "__version__",
]


def __getattr__(name: str):
    return getattr(zelt, name)
