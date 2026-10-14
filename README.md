# zelt ⚡

[![PyPI version](https://img.shields.io/pypi/v/zelt.svg)](https://pypi.org/project/zelt/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2610.zelt-b31b1b.svg)](paper/zelt_paper.tex)

**Calibrated, risk-controlled early exit for reasoning LLMs.**

Stops a reasoning model's chain-of-thought (CoT) once a learned decider is confident the answer is already correct — with a mathematically certified bound on potential accuracy loss ($\alpha \le 2\%$).

- **100% Local**: Runs on Apple Silicon MPS, CUDA, or CPU. No OpenAI/GPT API needed.
- **Pretrained & Calibrated**: Bundles a ready-to-use decider trained on real DeepSeek-R1 trajectories.
- **Zero Configuration**: Just import `zelt` and call.
- **Production-Ready**: Live streaming, batching, side-by-side comparison, dataset auditing, cost & carbon calculators, local web UI, Ollama/vLLM adapters, and Prometheus metrics.
- 📖 **Full Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md) for complete technical reference.
- 📄 **Research Paper**: Full LaTeX manuscript with Clopper-Pearson derivations available in [paper/zelt_paper.tex](paper/zelt_paper.tex).

---

## 🚀 Installation & PyPI

```bash
# Core package (benchmark, calibration, bundled decider, CPU):
pip install zelt-ai

# With local HuggingFace inference (DeepSeek-R1 on MPS / CUDA):
pip install "zelt-ai[model]"

# Full installation (server + web studio + ML):
pip install "zelt-ai[full]"

# For local development from source:
pip install -e .
```

---

## 💻 Python API Features

### 1. One-Liner Quickstart (`ask`)
```python
from zelt import ask

answer = ask("What is 17 * 23?")
print(answer)  # "391"
```

### 2. Side-by-Side Comparison (`compare`)
Compare Full unconstrained CoT vs. Zelt Early-Exit side-by-side:
```python
import zelt

comp = zelt.compare("What is 17 * 23?")
print(comp.summary())
```
**Output:**
```text
════════════════════════════════════════════════════════════
  Zelt Side-by-Side Comparison
────────────────────────────────────────────────────────────
  Question        : What is 17 * 23?
  Full CoT Answer : 391 (576 tokens, 2.10s)
  Zelt Answer     : 391 (144 tokens, 0.54s)
  Agreement       : MATCH (Exact)
  Tokens Saved    : 432 (75.0%)
  Effective Speedup: 3.89x faster
════════════════════════════════════════════════════════════
```

### 3. Wrap Functions with `@zelt.accelerate`
Automatically accelerate any reasoning function or query pipeline:
```python
import zelt


@zelt.accelerate
def reason(prompt: str) -> str:
    return "fallback"


print(reason("What is 15 * 15?"))  # "225" (exits early with saved compute)
```

### 4. Benchmark & Risk Audit (`evaluate`)
Run formal finite-sample risk evaluation with Clopper-Pearson bounds:
```python
import zelt

# Evaluates on real test trajectories and computes exact harm UCB
report = zelt.evaluate(limit=50)
print(report.summary())
```

### 5. Live Token Streaming (`stream`)
Stream reasoning tokens live with probe pulses and instant early exits:
```python
from zelt import stream

for chunk in stream("A train leaves Chicago travelling at 60 mph..."):
    if chunk.is_probe:
        print(f" [Probe gm={chunk.confidence:.2f}]", end="", flush=True)
    elif chunk.stopped:
        print(f"\n⚡ Stopped early at token {chunk.token_count}! Answer: {chunk.answer}")
        break
    else:
        print(chunk.text, end="", flush=True)
```

### 6. Batch Inference & Cost Tracking (`batch`)
Process queries in bulk with financial and environmental savings telemetry:
```python
from zelt import batch

questions = ["What is 17 * 23?", "Solve 3x + 9 = 24.", "What is 15% of 240?"]
results = batch(questions)
print(results.summary())
```

### 7. Model Presets (`list_presets`)
```python
import zelt

print(zelt.list_presets())
# Supports: DeepSeek-R1-Distill-Qwen (1.5B, 7B), DeepSeek-R1-Distill-Llama-8B, Qwen2.5-Math-7B
```

### 6. Integrating into Large LLM Systems

#### Option A: Inside Existing HuggingFace `model.generate(...)`
Attach Zelt directly to standard HuggingFace models via `StoppingCriteria`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from zelt import ZeltStoppingCriteria, StopDecider

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

criteria = ZeltStoppingCriteria(tokenizer, decider=StopDecider.load())

# Zelt interrupts generation the moment the answer is certified confident!
outputs = model.generate(
    input_ids, max_new_tokens=2048, stopping_criteria=StoppingCriteriaList([criteria])
)
```

#### Option B: Inside an API Service / FastAPI Backend
```python
from fastapi import FastAPI
from zelt import reduce_compute_thinking

app = FastAPI()


@app.post("/solve")
async def solve(prompt: str):
    # Replaces unconstrained 2,000-token CoT with risk-controlled early exit
    res = reduce_compute_thinking(prompt, verbose=False)
    return {
        "answer": str(res),
        "tokens_saved": res.tokens_saved,
        "saved_pct": res.saved_pct,
        "stopped_early": res.stopped,
    }
```

#### Option C: Inside AI Agent Frameworks (LangChain / AutoGen / CrewAI)
```python
from zelt import reduce_compute_thinking


class ReasoningTool:
    """Agent tool for complex math & logic sub-tasks without burning token budget."""

    def run(self, sub_question: str) -> str:
        return str(reduce_compute_thinking(sub_question, verbose=False))
```

---

## 🖥️ Interactive Web UI Dashboard

Launch a local visualizer with a live token stream, confidence gauges, and token comparison bars:

```bash
zelt ui --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 📊 Proven Results — Real DeepSeek-R1 on GSM8K (1,319 Real Trajectories)

Evaluated on held-out test split (331 questions) from real DeepSeek-R1-Distill-Qwen-1.5B runs:

| Method | Threshold ($\tau$) | Accuracy | $\Delta$ Acc | Avg Tokens | Token Reduction | Net (inc. probes) | Stop Rate | Harm Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full CoT (baseline)** | — | **78.25%** | baseline | 750.7 | 0.0% | 0.0% | 0.0% | 0.00% |
| Fixed Budget | 1,632 | 76.74% | -1.51% | 507.7 | 32.4% | 21.5% | 15.1% | 1.51% |
| Raw Confidence | 0.79 | 78.25% | +0.00% | 430.6 | 42.6% | 33.4% | 40.2% | 0.00% |
| **Learned Decider (zelt)** | **0.50** | **78.25%** | **+0.00%** | **465.9** | **37.9%** | **27.9%** | **37.8%** | **0.00%** |

- **37.9% compute reduction** (1.61× speedup)
- **Zero accuracy drop** (78.25% retained)
- **0.00% empirical harm** on test set (certified $\le 2\%$ at 90% confidence)
- **AUROC = 0.9976** and **ECE = 0.014** (3× better calibration than raw confidence)

---

## 🛠️ CLI Reference

| Command | Description |
|:---|:---|
| `zelt ui` | Launch local interactive web studio at `http://localhost:8000` |
| `zelt ask -q "..."` | Run early-exit inference on a single question |
| `zelt stream -q "..."` | Stream tokens live with intermediate probe confidence |
| `zelt batch -i questions.txt` | Run batch inference on a text file with cost metrics |
| `zelt benchmark` | Evaluate compute reduction on real bundled DeepSeek-R1 data |
| `zelt fit --inp <path>` | Train decider + calibrate threshold on custom runs |
| `zelt live-eval` | Live early-exit generation on held-out test split |
| `zelt demo -q "..."` | Side-by-side full vs early-exit comparison |
| `zelt selftest` | GPU-free pipeline check on synthetic data |

---

## 🔬 How It Works

```text
Prompt ──> DeepSeek-R1 (Local MPS / CUDA / Ollama)
                 │
                 ├── [Every 48 tokens] ──> Lightweight Probe (greedy decode ~10 tokens)
                 │                                │
                 │                     Extract: 18 signals (geometric-mean prob,
                 │                              entropy, answer stability,
                 │                              confidence acceleration)
                 │                                │
                 │                     Decider: P(correct | features)
                 │                                │
                 │               P(correct) ≥ τ ?
                 │              ┌───────┴───────┐
                 │             YES              NO
                 │              │                │
                 │         STOP EARLY ⚡     Keep thinking...
                 │    (Return probed ans)
```

1. **Periodic Probes**: Every $k$ tokens, a lightweight probe attempts greedy extraction of the intermediate answer.
2. **Feature Extraction**: Captures 18 signals including confidence geometric mean, sequence entropy, stability across consecutive probes, and confidence acceleration.
3. **Calibrated Decider**: Uses scikit-learn logistic regression with Clopper-Pearson risk control to find threshold $\tau$ bounding harm rate $\le \alpha$ (default 2%) with $(1-\delta)$ confidence (default 90%).
4. **Adaptive Early Stop**: Halts CoT generation the moment certainty is proven, saving ~40% tokens without harming answer quality.
