"""
Shared pytest fixtures for the zelt test suite.

All tests run without GPU / model weights (FallbackEngine is used automatically
when torch / transformers are not installed or no real model is available).
"""
import pytest
import zelt
from zelt import (
    EarlyStop,
    EarlyStopResult,
    BatchResult,
    CompareResult,
    EvaluationResult,
    StopDecider,
)


# ── Default instance fixture ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def zt() -> EarlyStop:
    """
    Session-scoped Zelt instance.  Shared across all tests to avoid re-loading
    the bundled decider and (optionally) model weights on every test.
    """
    return EarlyStop(verbose=False)


# ── Result fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def simple_result(zt) -> EarlyStopResult:
    """A real EarlyStopResult from the FallbackEngine on a trivial math query."""
    return zt.ask("What is 5 + 5?", verbose=False)


@pytest.fixture(scope="session")
def batch_result(zt) -> BatchResult:
    """A small batch of 3 questions."""
    return zt.batch(["What is 2 + 2?", "What is 3 * 4?", "What is 10 - 3?"], verbose=False)


@pytest.fixture(scope="session")
def compare_result(zt) -> CompareResult:
    """Side-by-side full CoT vs early exit."""
    return zt.compare("What is 10 + 20?", verbose=False)


@pytest.fixture(scope="session")
def eval_result(zt) -> EvaluationResult:
    """Evaluation on 10 bundled examples (fast)."""
    return zt.evaluate(limit=10, verbose=False)


# ── Decider fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bundled_decider() -> StopDecider:
    """Return the bundled pre-trained StopDecider."""
    import os
    path = os.path.join(os.path.dirname(zelt.__file__), "data", "pretrained_decider.joblib")
    return StopDecider.load(path)


# ── Synthetic trajectory fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_rows():
    """100 synthetic trajectory rows (no GPU required)."""
    from zelt.synthetic import make_synthetic
    return make_synthetic(100, seed=42)
