"""zelt: calibrated, risk-controlled early exit for reasoning LLMs.

This package is PEP 561 compliant (py.typed). Full type annotations are
included for mypy / pyright compatibility.

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

    # Server-side factories (no GPU needed locally)
    zt = zelt.from_ollama("deepseek-r1:1.5b")
    zt = zelt.from_vllm("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
"""

import logging as _logging

from .api      import EarlyStop, EarlyStopResult, StreamChunk, BatchResult, CompareResult, EvaluationResult
from .answers  import extract_answer, is_correct
from .decider  import StopDecider, make_stop_fn, featurize, fit_and_report

# Primary aliases for zelt
Zelt = EarlyStop
ZeltResult = EarlyStopResult

__version__ = "0.2.1"

# Library-level logger — users can configure via logging.getLogger('zelt')
_log = _logging.getLogger(__name__)
if not _log.handlers:
    _log.addHandler(_logging.NullHandler())

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


def compare(question: str, **kwargs) -> CompareResult:
    """Run Full CoT vs. Zelt Early-Exit side-by-side with diff and token savings."""
    return _get_default().compare(question, **kwargs)


def evaluate(dataset=None, **kwargs) -> EvaluationResult:
    """Evaluate benchmark dataset with exact Clopper-Pearson risk bounds."""
    return _get_default().evaluate(dataset=dataset, **kwargs)


def calibrate(data=None, alpha: float = 0.02, delta: float = 0.10, outdir: str = "results") -> dict:
    """Calibrate stopping threshold on trajectory data with certified risk bound."""
    return _get_default().calibrate(data=data, alpha=alpha, delta=delta, outdir=outdir)


def accelerate(func=None, **kwargs):
    """Decorator to wrap reasoning LLM calls with automated early stopping."""
    def decorator(fn):
        def wrapper(*args, **fn_kwargs):
            if args and isinstance(args[0], str):
                return ask(args[0], **kwargs)
            return fn(*args, **fn_kwargs)
        wrapper.__name__ = getattr(fn, "__name__", "accelerate_wrapper")
        wrapper.__doc__ = getattr(fn, "__doc__", "")
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator


def from_ollama(
    model: str = "deepseek-r1:1.5b",
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> EarlyStop:
    """
    Factory: create a Zelt instance backed by a local Ollama server.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. 'deepseek-r1:1.5b' or 'qwen2.5:7b'.
    base_url : str
        Ollama server base URL (default: http://localhost:11434).
    **kwargs
        Additional keyword arguments forwarded to `Zelt.__init__`
        (e.g. alpha, delta, verbose).

    Returns
    -------
    EarlyStop
        Fully configured Zelt instance connected to the Ollama backend.

    Example
    -------
    >>> import zelt
    >>> zt = zelt.from_ollama("deepseek-r1:1.5b")
    >>> print(zt.ask("What is 12 * 12?"))
    """
    return EarlyStop(model=model, backend="ollama", base_url=base_url, **kwargs)


def from_vllm(
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    base_url: str = "http://localhost:8000",
    **kwargs,
) -> EarlyStop:
    """
    Factory: create a Zelt instance backed by a vLLM / OpenAI-compatible server.

    Parameters
    ----------
    model : str
        Model ID as served by vLLM.
    base_url : str
        vLLM server base URL (default: http://localhost:8000).
    **kwargs
        Additional keyword arguments forwarded to `Zelt.__init__`.

    Returns
    -------
    EarlyStop
        Fully configured Zelt instance connected to the vLLM backend.

    Example
    -------
    >>> import zelt
    >>> zt = zelt.from_vllm("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ...                      base_url="http://gpu-server:8000")
    >>> print(zt.ask("Solve: 3x + 9 = 24"))
    """
    return EarlyStop(model=model, backend="vllm", base_url=base_url, **kwargs)


def token_budget(
    max_tokens: int,
    question: str,
    **kwargs,
) -> EarlyStopResult:
    """
    Run inference with a hard token budget: stop as soon as a valid answer
    is found within `max_tokens` output tokens, regardless of confidence.

    This is a deterministic fallback — useful for strict latency SLAs where
    you'd rather return a best-guess answer than time out.

    Parameters
    ----------
    max_tokens : int
        Maximum number of output tokens to generate.
    question : str
        The reasoning prompt.
    **kwargs
        Extra kwargs forwarded to `Zelt.__init__` (e.g. model, backend).

    Returns
    -------
    EarlyStopResult

    Example
    -------
    >>> import zelt
    >>> res = zelt.token_budget(200, "What is 17 * 23?")
    >>> print(res)  # answer extracted within budget
    """
    from .decider import StopDecider, make_stop_fn
    import os

    _BUNDLED_DECIDER = os.path.join(os.path.dirname(__file__), "data", "pretrained_decider.joblib")
    zt = EarlyStop(**kwargs)

    # Override stop function: stop when we have an answer, regardless of tau
    def budget_stop_fn(probes):
        if not probes:
            return False
        last = probes[-1]
        # Stop if we have an answer and tokens already exceeded budget
        return last.get("ans") is not None and last.get("t", 0) >= max_tokens

    import time
    zt._load_engine()
    t0 = time.time()
    out = zt._engine.run(question, stop_fn=budget_stop_fn)
    elapsed = time.time() - t0
    full_est = out["tokens"] if not out["stopped"] else int(out["tokens"] / 0.62)
    scores = None
    if out["probes"] and zt._decider:
        scores = list(zt._decider.score(out["probes"]))
    return EarlyStopResult(
        question=question,
        answer=out["answer"],
        tokens=out["tokens"],
        full_est=full_est,
        stopped=out["stopped"],
        probes=out["probes"],
        scores=scores,
        elapsed=elapsed,
    )


def risk_report(
    data=None,
    alpha: float = 0.02,
    delta: float = 0.10,
    verbose: bool = True,
) -> dict:
    """
    Compute a certified Clopper-Pearson risk audit from pre-recorded trajectories.
    Does NOT require model inference — works on any saved JSONL trajectory file.

    Parameters
    ----------
    data : str | list | None
        Path to a `.jsonl` trajectory file, a list of trajectory dicts,
        or None to use the bundled real DeepSeek-R1 GSM8K trajectories.
    alpha : float
        Maximum permitted harm rate (default 0.02 = 2%).
    delta : float
        Confidence failure probability (default 0.10 = 90% confidence).
    verbose : bool
        Print the formatted risk report table.

    Returns
    -------
    dict
        Full report dict with keys: n_train, n_cal, n_test, methods, calibration.

    Example
    -------
    >>> import zelt
    >>> report = zelt.risk_report()              # uses bundled GSM8K data
    >>> report = zelt.risk_report("my_runs.jsonl")  # your own trajectories
    """
    import os
    from .decider import load_rows, fit_and_report

    _BUNDLED_DATA = os.path.join(os.path.dirname(__file__), "data", "gsm8k_real_deepseek_r1.jsonl")

    if data is None:
        data = _BUNDLED_DATA
    if isinstance(data, str):
        rows = load_rows(data)
    else:
        rows = list(data)

    import tempfile
    outdir = tempfile.mkdtemp(prefix="zelt_risk_report_")
    return fit_and_report(rows, outdir=outdir, alpha=alpha, delta=delta, verbose=verbose)


def list_presets() -> dict:
    """Return dictionary of supported reasoning models and probe presets."""
    return {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "chunk": 48,
            "min_tokens": 48,
            "probe_max": 16,
            "tested_accuracy": 0.7825,
            "avg_token_reduction": 0.379,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "chunk": 48,
            "min_tokens": 48,
            "probe_max": 16,
            "tested_accuracy": 0.892,
            "avg_token_reduction": 0.365,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
            "chunk": 48,
            "min_tokens": 48,
            "probe_max": 16,
            "tested_accuracy": 0.875,
            "avg_token_reduction": 0.352,
        },
        "Qwen/Qwen2.5-Math-7B-Instruct": {
            "chunk": 40,
            "min_tokens": 40,
            "probe_max": 16,
            "tested_accuracy": 0.841,
            "avg_token_reduction": 0.334,
        },
    }


__all__ = [
    # ── primary API ──
    "reduce_compute_thinking",
    "ask",
    "stream",
    "batch",
    "compare",
    "evaluate",
    "calibrate",
    "accelerate",
    "list_presets",
    # ── factory / convenience ──
    "from_ollama",
    "from_vllm",
    "token_budget",
    "risk_report",
    # ── types ──
    "Zelt",
    "ZeltResult",
    "EarlyStop",
    "EarlyStopResult",
    "StreamChunk",
    "BatchResult",
    "CompareResult",
    "EvaluationResult",
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
