"""
Tests for new production-grade factory functions added in v0.2.1:
from_ollama, from_vllm, token_budget, risk_report
"""
import pytest
import zelt
from zelt import EarlyStop, EarlyStopResult


class TestFromOllama:
    def test_returns_earlystop_instance(self):
        zt = zelt.from_ollama("deepseek-r1:1.5b", verbose=False)
        assert isinstance(zt, EarlyStop)

    def test_backend_is_ollama(self):
        zt = zelt.from_ollama("deepseek-r1:1.5b", verbose=False)
        assert zt.backend == "ollama"

    def test_model_set_correctly(self):
        zt = zelt.from_ollama("qwen2.5:7b", verbose=False)
        assert zt._model_name == "qwen2.5:7b"

    def test_custom_base_url(self):
        zt = zelt.from_ollama(base_url="http://192.168.1.100:11434", verbose=False)
        assert zt.base_url == "http://192.168.1.100:11434"

    def test_extra_kwargs_passed(self):
        zt = zelt.from_ollama(alpha=0.05, delta=0.05, verbose=False)
        assert zt._alpha == 0.05
        assert zt._delta == 0.05


class TestFromVLLM:
    def test_returns_earlystop_instance(self):
        zt = zelt.from_vllm(verbose=False)
        assert isinstance(zt, EarlyStop)

    def test_backend_is_vllm(self):
        zt = zelt.from_vllm(verbose=False)
        assert zt.backend == "vllm"

    def test_model_set_correctly(self):
        zt = zelt.from_vllm("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", verbose=False)
        assert zt._model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def test_custom_base_url(self):
        zt = zelt.from_vllm(base_url="http://gpu-host:8000", verbose=False)
        assert zt.base_url == "http://gpu-host:8000"


class TestTokenBudget:
    def test_returns_earlystop_result(self):
        res = zelt.token_budget(100, "What is 5 + 5?")
        assert isinstance(res, EarlyStopResult)

    def test_answer_not_none(self):
        res = zelt.token_budget(200, "What is 12 * 12?")
        assert res.answer is not None

    def test_tokens_not_exceeds_budget_by_much(self):
        # Token budget is soft (enforced at probe boundaries), so allow up to 2x chunk overhead
        res = zelt.token_budget(150, "What is 7 + 7?")
        assert res.tokens <= 500  # well under full CoT

    def test_has_elapsed(self):
        res = zelt.token_budget(200, "What is 5 * 5?")
        assert res.elapsed >= 0.0


class TestRiskReport:
    def test_returns_dict(self):
        rep = zelt.risk_report(verbose=False)
        assert isinstance(rep, dict)

    def test_has_required_keys(self):
        rep = zelt.risk_report(verbose=False)
        assert "n_train" in rep
        assert "n_cal" in rep
        assert "n_test" in rep
        assert "methods" in rep
        assert "calibration" in rep

    def test_learned_method_present(self):
        rep = zelt.risk_report(verbose=False)
        assert "learned" in rep["methods"]

    def test_custom_alpha(self):
        rep = zelt.risk_report(alpha=0.05, verbose=False)
        assert rep["alpha"] == 0.05

    def test_calibration_metrics_present(self):
        rep = zelt.risk_report(verbose=False)
        cal = rep["calibration"]
        assert "learned" in cal
        assert "ece" in cal["learned"]
        assert "auroc" in cal["learned"]


class TestFactoriesInAll:
    """Ensure all new functions are exported in __all__."""
    def test_from_ollama_in_all(self):
        assert "from_ollama" in zelt.__all__

    def test_from_vllm_in_all(self):
        assert "from_vllm" in zelt.__all__

    def test_token_budget_in_all(self):
        assert "token_budget" in zelt.__all__

    def test_risk_report_in_all(self):
        assert "risk_report" in zelt.__all__
