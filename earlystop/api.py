"""Forward from zelt.api for backward compatibility."""
from zelt.api import (
    EarlyStop,
    EarlyStopResult,
    Zelt,
    ZeltResult,
    StreamChunk,
    BatchResult,
)

__all__ = [
    "EarlyStop",
    "EarlyStopResult",
    "Zelt",
    "ZeltResult",
    "StreamChunk",
    "BatchResult",
]
