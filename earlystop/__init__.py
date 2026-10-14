"""Forward from zelt for backward compatibility."""
from zelt import (
    reduce_compute_thinking,
    ask,
    stream,
    batch,
    Zelt,
    ZeltResult,
    EarlyStop,
    EarlyStopResult,
    StreamChunk,
    BatchResult,
    StopDecider,
    make_stop_fn,
    featurize,
    fit_and_report,
    extract_answer,
    is_correct,
)

__version__ = "0.2.0"

__all__ = [
    "reduce_compute_thinking",
    "ask",
    "stream",
    "batch",
    "Zelt",
    "ZeltResult",
    "EarlyStop",
    "EarlyStopResult",
    "StreamChunk",
    "BatchResult",
    "EarlyStopper",
    "StopDecider",
    "make_stop_fn",
    "featurize",
    "fit_and_report",
    "extract_answer",
    "is_correct",
]


def __getattr__(name):
    if name == "EarlyStopper":
        from .core import EarlyStopper
        return EarlyStopper
    raise AttributeError(f"module 'earlystop' has no attribute {name!r}")
