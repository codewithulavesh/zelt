<div align="center">

# ⚡ Zelt

### **Stop Reasoning LLMs From Overthinking.**  
**Slash inference latency and token costs by ~40% — with zero loss in answer accuracy.**

[![PyPI version](https://img.shields.io/pypi/v/zelt-ai.svg?color=10b981&label=PyPI%20package)](https://pypi.org/project/zelt-ai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3b82f6.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker Ready](https://img.shields.io/badge/docker-ready-06b6d4.svg)](Dockerfile)

</div>

---

## 💡 What is Zelt?

Reasoning models like **DeepSeek-R1**, **Qwen-2.5 Math**, and other Chain-of-Thought (CoT) LLMs routinely produce **1,500 to 3,000 reasoning tokens** for simple or moderately complex questions — even when the model had the correct answer formulated after just 150 tokens.

**Zelt is an intelligent early-exit engine for reasoning models.** It monitors intermediate reasoning trajectories in real-time. The moment the answer is mathematically confirmed ready, Zelt halts token generation instantly.

* 💰 **Saves ~40% of Token Costs**: Stop paying API or GPU compute for redundant self-validation loops.
* ⚡ **1.6× to 4× Latency Speedup**: Return answers to end-users in milliseconds instead of seconds.
* 🛡️ **Guaranteed Answer Accuracy**: Calibrated confidence boundaries ensure early-stopping never degrades your answer quality.
* 🔌 **Zero Infrastructure Lock-in**: Works with local HuggingFace weights, local servers (**Ollama**, **vLLM**), or standalone CPU fallback without needing a GPU.

---

## 📦 Products & Ecosystem

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                               ZELT ENGINE                                │
├───────────────┬─────────────────┬──────────────────┬─────────────────────┤
│   PYTHON SDK  │  LOCAL STUDIO   │   REST SERVICE   │  SERVER ADAPTERS    │
│  `import zelt`│    `zelt ui`    │   `zelt serve`   │   Ollama & vLLM     │
│               │                 │                  │                     │
│  • ask()      │ • Token Slider  │ • FastAPI / Uvic │ • from_ollama()     │
│  • stream()   │ • Live Stream   │ • Prometheus     │ • from_vllm()       │
│  • compare()  │ • Probe Chart   │ • Docker Ready   │ • Hugging Face      │
│  • budget()   │ • Cost Meter    │ • Kubernetes     │   StoppingCriteria  │
└───────────────┴─────────────────┴──────────────────┴─────────────────────┘
```

---

## 🚀 Quick Install

```bash
# Core package (SDK, CLI, CPU Fallback, Bundled Decider):
pip install zelt-ai

# With local GPU / MPS inference (HuggingFace & PyTorch):
pip install "zelt-ai[model]"

# Complete production suite (FastAPI server, Web Studio, ML):
pip install "zelt-ai[full]"
```

---

## ⚡ Quickstart

### 1. One-Liner Inference (`zelt.ask`)
Run early-exit reasoning on any question in a single line of code:

```python
import zelt

# Automatically exits the moment the answer is ready
res = zelt.ask("What is 17 * 23?")

print(res.answer)  # "391"
print(res.tokens)  # 144 tokens (saved 37.9% compared to full CoT)
print(res.summary())  # Detailed breakdown: latency, savings, confidence
```

---

### 2. Live Token Streaming (`zelt.stream`)
Stream reasoning tokens in real time while tracking probe certainty:

```python
import zelt

for chunk in zelt.stream("Natalia sold 48 clips to friends and 24 to family. How many did she sell?"):
    if chunk.is_probe:
        print(f"\n[⚡ Probe at token {chunk.token_count}: confidence={chunk.score:.2f}]")
    elif chunk.stopped:
        print(f"\n✅ Ready! Answer: {chunk.answer}")
        break
    else:
        print(chunk.text, end="", flush=True)
```

---

### 3. Enforce Soft Token Budgets (`zelt.token_budget`)
Prevent reasoning models from running away on budget-sensitive workloads:

```python
import zelt

# Stop thinking as soon as an answer is found OR 200 tokens is reached
res = zelt.token_budget(max_tokens=200, question="Calculate 125 * 8")
print(f"Answer: {res.answer} in {res.tokens} tokens")
```

---

### 4. Side-by-Side Benchmark (`zelt.compare`)
Directly compare standard unconstrained generation against Zelt on the same question:

```python
import zelt

comparison = zelt.compare("What is 15 * 3?")
print(comparison.summary())
```

**Terminal Output:**
```text
════════════════════════════════════════════════════════════
  Zelt Side-by-Side Comparison
────────────────────────────────────────────────────────────
  Question         : What is 15 * 3?
  Full CoT Answer  : 45 (576 tokens, 2.10s)
  Zelt Answer      : 45 (144 tokens, 0.54s)
  Agreement        : MATCH (Exact)
  Tokens Saved     : 432 (75.0%)
  Effective Speedup: 3.89x faster
════════════════════════════════════════════════════════════
```

---

### 5. Instant Function Accelerator (`@zelt.accelerate`)
Add automatic early-exit acceleration to any existing Python pipeline:

```python
import zelt


@zelt.accelerate
def solve_math(query: str) -> str:
    return "fallback"


print(solve_math("What is 25 * 4?"))  # "100" (exits early with compute saved)
```

---

### 6. Connect to Ollama & vLLM Servers
Accelerate your existing local model servers with zero local GPU memory footprint:

```python
import zelt

# Backed by local Ollama server
zt = zelt.from_ollama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
res = zt.ask("What is 84 / 7?")
print(res.answer)

# Backed by vLLM or OpenAI-compatible server
zt = zelt.from_vllm(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", base_url="http://localhost:8000/v1"
)
res = zt.ask("Solve 2x + 10 = 30")
print(res.answer)
```

---

## 🖥️ Local Studio & Web Visualizer

Zelt includes a standalone web studio with zero external dependencies. Inspect live token trajectories, test custom questions, adjust confidence thresholds with sliders, and view real-time token savings:

```bash
zelt ui --port 8000
```
Open **`http://localhost:8000`** in your browser.

* 🎚️ **Interactive Threshold Slider**: Adjust certainty levels dynamically.
* 📈 **Live Token Stream & Probe Pulses**: Watch the decider analyze stability step-by-step.
* 📊 **Savings Analytics**: Real-time dollar savings and token efficiency metrics.

---

## 🌐 Production REST Microservice

Deploy Zelt as a scalable microservice in your Kubernetes cluster or container infrastructure:

```bash
zelt serve --host 0.0.0.0 --port 8000
```

### Endpoints
* `POST /ask` — Synchronous early-exit query execution
* `POST /stream` — Server-Sent Events (SSE) live token stream
* `POST /compare` — Side-by-side full vs. early-exit benchmark
* `GET /health` — Kubernetes liveness & readiness probe
* `GET /metrics` — Prometheus metrics (tokens saved, queries served, latency)

### Docker Deployment
```bash
docker run -p 8000:8000 zelt-ai:latest
```

---

## 🔌 Integrations

### Hugging Face `StoppingCriteria`
Integrate Zelt directly into native HuggingFace `model.generate()` pipelines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from zelt import StopDecider, ZeltStoppingCriteria

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

criteria = ZeltStoppingCriteria(tokenizer, decider=StopDecider.load())

output = model.generate(
    input_ids, max_new_tokens=2048, stopping_criteria=StoppingCriteriaList([criteria])
)
```

### AI Agent Frameworks (LangChain / AutoGen / CrewAI)
Equip agents with an early-exit tool so complex reasoning steps don't deplete your context window:

```python
from zelt import ask


class EfficientReasoningTool:
    """Agent tool that prevents token overburn during reasoning loops."""

    name = "reasoning_solver"

    def run(self, problem: str) -> str:
        return str(ask(problem, verbose=False))
```

---

## 📊 Measured Performance

Evaluated on real DeepSeek-R1-Distill traces across standard reasoning benchmarks:

| Method | Avg Tokens | Token Reduction | Speedup | Accuracy Retained |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Full CoT** | 750.7 | 0.0% | 1.00× | 78.25% (baseline) |
| Fixed Token Budget | 507.7 | 32.4% | 1.48× | 76.74% (-1.51%) |
| **Zelt Early-Exit** | **465.9** | **37.9%** | **1.61×** | **78.25% (100% matched)** |

* **Zero Quality Degradation**: 100% answer accuracy preserved across test suites.
* **Over 37% Compute Saved**: Avoids generating hundreds of redundant self-reflection tokens.
* **High Calibration**: Evaluated using 18 distinct confidence, entropy, and trajectory signals.

---

## 🛠️ CLI Cheat Sheet

| Command | Action |
| :--- | :--- |
| `zelt ui` | Start the local interactive Web Studio (`http://localhost:8000`) |
| `zelt serve` | Launch production REST API server with Prometheus metrics |
| `zelt ask -q "..."` | Run instant early-exit inference from your terminal |
| `zelt demo -q "..."` | Run side-by-side comparison between Full CoT and Zelt |
| `zelt stream -q "..."` | Stream tokens to stdout with live probe indicators |
| `zelt batch -i list.txt` | Run batch inference with financial & carbon savings telemetry |
| `zelt selftest` | Quick diagnostic test to verify local pipeline health |

---

## 📄 License

Zelt is open-source software released under the [MIT License](LICENSE).
