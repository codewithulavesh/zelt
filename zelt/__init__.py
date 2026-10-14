"""zelt: calibrated, risk-controlled early exit for reasoning LLMs.

Quickstart (zero config)
------------------------
    from zelt import Zelt, ask, stream, batch

    # One-liner
    answer = ask("What is 17 * 23?")
    print(answer)                           # "391"

    # Streaming
    for chunk in stream("What is 17 * 23?"):
        print(chunk.text, end="", flush=True)

    # Batch with cost tracking
    results = batch(["What is 2+2?", "What is 3*5?"])
    print(results.summary())

    # Object API
    zt = Zelt()                             # auto-loads bundled decider
    result = zt.ask("What is 17 * 23?")     # ZeltResult
    print(result.summary())                 # token savings + confidence stats
"""

from .api      import EarlyStop, EarlyStopResult, StreamChunk, BatchResult
from .answers  import extract_answer, is_correct
from .decider  import StopDecider, make_stop_fn, featurize, fit_and_report

# Primary aliases for zelt
Zelt = EarlyStop
ZeltResult = EarlyStopResult

__version__ = "0.2.0"

_default_instance = None


def _get_default():
    global _default_instance
    if _default_instance is None:
        _default_instance = Zelt()
    return _default_instance


def ask(question: str, **kwargs) -> EarlyStopResult:
    """Zero-config one-liner: ask a question with early stopping."""
    return _get_default().ask(question, **kwargs)


def stream(question: str):
    """Zero-config streaming: stream tokens with early exit."""
    return _get_default().stream(question)


def batch(questions, **kwargs) -> BatchResult:
    """Zero-config batching: process list of questions with savings metrics."""
    return _get_default().batch(questions, **kwargs)


def reduce_compute_thinking(prompt: str, **kwargs) -> EarlyStopResult:
    """Halt reasoning overthinking to drastically reduce test-time compute."""
    return _get_default().reduce_compute_thinking(prompt, **kwargs)


__all__ = [
    # ── primary API ──
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
    # ── power-user / internal API ──
    "ZeltStoppingCriteria",
    "EarlyStopper",
    "StopDecider",
    "make_stop_fn",
    "featurize",
    "fit_and_report",
    "extract_answer",
    "is_correct",
]


def __getattr__(name):
    """Lazy-load EarlyStopper / ZeltStoppingCriteria (needs torch)."""
    if name == "EarlyStopper":
        from .core import EarlyStopper
        return EarlyStopper
    if name == "ZeltStoppingCriteria":
        from .core import ZeltStoppingCriteria
        return ZeltStoppingCriteria
    raise AttributeError(f"module 'zelt' has no attribute {name!r}")
