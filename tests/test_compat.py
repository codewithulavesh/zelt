"""
Tests for backward-compatibility shim (earlystop package).
Ensures existing code using earlystop continues to work seamlessly.
"""

import earlystop
import zelt
from earlystop import (
    EarlyStop,
    EarlyStopper,
    EarlyStopResult,
    StopDecider,
    extract_answer,
    is_correct,
)
from earlystop.adapters import OllamaAdapter, VLLMAdapter
from earlystop.answers import extract_answer as ea_ans
from earlystop.core import EarlyStopper as es_core
from earlystop.decider import StopDecider as sd_dec


def test_earlystop_version():
    assert earlystop.__version__ == zelt.__version__


def test_earlystop_ask():
    res = earlystop.ask("What is 10 + 15?")
    assert str(res) == "25"
    assert isinstance(res, EarlyStopResult)


def test_earlystop_class():
    zt = EarlyStop(verbose=False)
    res = zt.ask("What is 8 * 8?")
    assert str(res) == "64"


def test_earlystop_answers():
    assert extract_answer("Result is \\boxed{99}") == "99"
    assert ea_ans("Result is \\boxed{99}") == "99"
    assert is_correct("99", "99")


def test_earlystop_core_import():
    assert es_core is EarlyStopper


def test_earlystop_decider_import():
    assert sd_dec is StopDecider


def test_earlystop_adapters_import():
    assert OllamaAdapter is zelt.adapters.OllamaAdapter
    assert VLLMAdapter is zelt.adapters.VLLMAdapter


def test_earlystop_getattr():
    assert hasattr(earlystop, "token_budget")
    assert hasattr(earlystop, "risk_report")
