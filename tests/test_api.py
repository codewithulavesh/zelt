import zelt


def test_ask():
    res = zelt.ask("What is 17 * 23?")
    assert str(res) == "391"
    assert res.tokens > 0
    assert res.cost_saved() >= 0.0
    assert res.carbon_saved() >= 0.0
    summary = res.summary()
    assert "Answer" in summary
    assert "Tokens" in summary


def test_compare():
    comp = zelt.compare("What is 10 + 20?", verbose=False)
    assert comp.match is True
    assert comp.tokens_saved >= 0
    assert comp.speedup >= 1.0
    summary = comp.summary()
    assert "Side-by-Side Comparison" in summary


def test_batch():
    questions = ["What is 5 + 5?", "What is 10 * 10?"]
    batch_res = zelt.batch(questions, verbose=False)
    assert batch_res.n == 2
    assert batch_res.speedup >= 1.0
    assert "zelt_queries_total 2" in batch_res.export_prometheus()


def test_accelerate_decorator():
    @zelt.accelerate
    def calculate(q: str):
        return "fallback"

    ans = calculate("What is 12 + 12?")
    assert str(ans) == "24"


def test_evaluate_bundled():
    # Evaluate limit of 10 examples
    report = zelt.evaluate(limit=10, verbose=False)
    assert report.n == 10
    assert 0.0 <= report.harm_rate <= 1.0
    assert report.ucb <= 1.0
    assert "Benchmark & Risk Audit Report" in report.summary()


def test_list_presets():
    presets = zelt.list_presets()
    assert "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" in presets
    assert presets["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]["chunk"] == 48
