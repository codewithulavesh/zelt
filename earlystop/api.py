"""Forward from zelt.api for backward compatibility."""

from zelt.api import (
    BatchResult,
    EarlyStop,
    EarlyStopResult,
    StreamChunk,
    Zelt,
    ZeltResult,
)

__all__ = [
    "EarlyStop",
    "EarlyStopResult",
    "Zelt",
    "ZeltResult",
    "StreamChunk",
    "BatchResult",
]
