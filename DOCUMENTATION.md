# Zelt ⚡ Comprehensive Technical Documentation

> **Calibrated, Risk-Controlled Early Exit for Reasoning Large Language Models**  
> *Version: 0.2.0*  
> *License: MIT*  

---

## Table of Contents
1. [Introduction & Motivation](#1-introduction--motivation)
2. [Architecture & How It Works](#2-architecture--how-it-works)
3. [Installation & Requirements](#3-installation--requirements)
4. [Quickstart Guides](#4-quickstart-guides)
   - [One-Liner Quickstart](#one-liner-quickstart)
   - [Object-Oriented API](#object-oriented-api)
   - [Real-Time Streaming](#real-time-streaming)
   - [Batch Processing & Cost Calculator](#batch-processing--cost-calculator)
   - [Interactive Web Studio](#interactive-web-studio)
5. [Complete API Reference](#5-complete-api-reference)
   - [Module-Level Functions](#module-level-functions)
   - [Class `Zelt`](#class-zelt-alias-earlystop)
   - [Class `EarlyStopResult`](#class-earlystopresult-alias-zeltresult)
   - [Class `StreamChunk`](#class-streamchunk)
   - [Class `BatchResult`](#class-batchresult)
   - [Class `StopDecider`](#class-stopdecider)
   - [Adapters (`OllamaAdapter`, `VLLMAdapter`)](#adapters)
6. [Mathematical Foundations & Risk Control](#6-mathematical-foundations--risk-control)
   - [Definition of Harm](#definition-of-harm)
   - [Clopper-Pearson Upper Confidence Bound](#clopper-pearson-upper-confidence-bound)
   - [Fixed-Sequence Hypothesis Testing](#fixed-sequence-hypothesis-testing)
7. [Empirical Benchmark Results](#7-empirical-benchmark-results)
8. [CLI Reference Manual](#8-cli-reference-manual)
9. [Custom Model Training & Calibration](#9-custom-model-training--calibration)
10. [Observability & Prometheus Integration](#10-observability--prometheus-integration)

---

## 1. Introduction & Motivation

Reasoning models like **DeepSeek-R1**, **OpenAI o1/o3**, and **Qwen-2.5-Math** solve complex problems by generating extended chains-of-thought (CoT). While powerful, they exhibit a major inefficiency: **overthinking**.

For easy and moderately difficult queries, the model often arrives at the correct answer within the first 100–300 tokens, but continues generating hundreds of superfluous tokens (e.g., repeatedly re-verifying, stating the same calculations, or second-guessing itself).

### The Solution: `zelt`
`zelt` periodically tests intermediate states with lightweight, non-invasive greedy probes and extracts an 18-dimensional feature vector measuring confidence, sequence entropy, and answer stability. A calibrated statistical decider halts generation the exact moment the answer is proven confident, providing a **provable upper bound on accuracy loss** ($\alpha \le 2\%$) at 90% confidence.

- **Token Reduction**: 37.9% average reduction on real DeepSeek-R1 traces (up to 65% on easy problems).
- **Inference Speedup**: 1.61× wall-clock acceleration.
- **Accuracy Preserved**: 0.00% drop on held-out test splits.
- **100% Local**: No external API keys or GPT calls needed.

---

## 2. Architecture & How It Works

```text
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  DeepSeek-R1 (Local Engine: PyTorch / MPS / CUDA / Ollama)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                [Every chunk = 48 tokens]
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Lightweight Intermediate Probe                             │
│  - Executes 10-token greedy decode on current KV cache      │
│  - Extracts candidate answer text                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Extraction (18 Signals)                            │
│  - Geometric-mean token probability                         │
│  - Sequence entropy (uncertainty)                           │
│  - Answer stability across consecutive probes               │
│  - Confidence acceleration (2nd derivative)                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Calibrated Decider (Risk-Controlled Logistic Classifier)   │
│  Computes P(Correct | Features)                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                       Is P(Correct) ≥ τ ?
                      ┌────────┴────────┐
                     YES                NO
                      │                  │
                      ▼                  ▼
             ⚡ EARLY STOP           Continue CoT
             Return Answer           Generation
```

---

## 3. Installation & Requirements

### System Requirements
- **Python**: `>= 3.9`
- **Supported Accelerators**: Apple Silicon (MPS), NVIDIA CUDA, or CPU.

### Basic Installation (Evaluation & Decider Fitting)
Runs on CPU in seconds; requires zero heavy ML dependencies:
```bash
pip install -e .
```

### Full Installation (with Local HuggingFace Model Weights)
Installs PyTorch and HuggingFace Transformers:
```bash
pip install -e ".[model]"
```

---

## 4. Quickstart Guides

### Reduce Compute & Thinking Time (`reduce_compute_thinking`)
```python
from zelt import reduce_compute_thinking

# Primary compute-saving entrypoint
result = reduce_compute_thinking("What is 17 * 23?")
print("Answer:", result)            # "391"
print(result.summary())             # Displays tokens saved, speedup, and cost savings
```

### One-Liner Quickstart (`ask`)
```python
from zelt import ask

# Automatically loads the bundled certified decider + model
answer = ask("What is 17 * 23?")
print(answer)  # "391"
```

### Object-Oriented API
```python
from zelt import Zelt

zt = Zelt()  # Uses bundled DeepSeek-R1 decider with harm bound ≤ 2%

result = zt.ask("Natalia sold clips to 48 friends and 24 family members. How many clips did she sell?")
print(result)            # "72"
print(result.summary())  # Detailed breakdown of tokens saved and confidence
```

### Real-Time Streaming
```python
from zelt import stream

for chunk in stream("A car travels 60 miles per hour for 3.5 hours. How far does it travel?"):
    if chunk.is_probe:
        print(f" [Probe gm={chunk.confidence:.2f}]", end="", flush=True)
    elif chunk.stopped:
        print(f"\n⚡ Early exit triggered at token {chunk.token_count}! Answer: {chunk.answer}")
        break
    else:
        print(chunk.text, end="", flush=True)
```

### Batch Processing & Cost Calculator
```python
from zelt import batch

questions = [
    "What is 15% of 240?",
    "If x + 5 = 12, what is 2x?",
    "How many days are in 6 weeks?"
]

results = batch(questions)
print(results.summary())
print(f"Total cost saved: ${results.cost_saved():.5f}")
print(f"Total CO2 avoided: {results.carbon_saved():.2f} g")
```

### Interactive Web Studio
Launch the browser dashboard with live token streaming and probe inspectors:
```bash
zelt ui --port 8000
```
Then navigate to `http://localhost:8000`.

### Production Architecture Integration Patterns

#### 1. Inside HuggingFace `model.generate(...)` via `StoppingCriteria`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from zelt import ZeltStoppingCriteria, StopDecider

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

criteria = ZeltStoppingCriteria(tokenizer, decider=StopDecider.load())

# Standard HuggingFace generation loop - Zelt halts the forward passes dynamically
outputs = model.generate(
    inputs,
    max_new_tokens=2048,
    stopping_criteria=StoppingCriteriaList([criteria])
)
```

#### 2. Inside a Microservice / FastAPI Backend
```python
from fastapi import FastAPI
from pydantic import BaseModel
from zelt import reduce_compute_thinking

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/v1/reason")
async def handle_reason(query: Query):
    result = reduce_compute_thinking(query.prompt, verbose=False)
    return {
        "answer": str(result),
        "tokens_used": result.tokens,
        "tokens_saved": result.tokens_saved,
        "compute_saved_percent": result.saved_pct,
        "stopped_early": result.stopped,
        "cost_saved_usd": result.cost_saved()
    }
```

#### 3. Inside Agent Loops (LangChain, AutoGen, CrewAI)
```python
from zelt import reduce_compute_thinking

def reasoning_agent_tool(task_description: str) -> str:
    """Executes complex reasoning sub-tasks with bounded test-time compute."""
    result = reduce_compute_thinking(task_description, verbose=False)
    return str(result)
```

---

## 5. Complete API Reference

### Module-Level Functions

#### `zelt.ask(question: str, **kwargs) -> EarlyStopResult`
Executes early-exit inference on a single query using a shared singleton engine.
- **Parameters**:
  - `question` *(str)*: Natural language problem.
- **Returns**: `EarlyStopResult` object.

#### `zelt.stream(question: str) -> Generator[StreamChunk, None, EarlyStopResult]`
Yields tokens and probe events in real-time.

#### `zelt.batch(questions: list[str], **kwargs) -> BatchResult`
Runs batch inference across multiple prompts and computes aggregate telemetry.

#### `zelt.extract_answer(text: str) -> str | None`
Extracts final numeric or symbolic answers from raw reasoning text (handles LaTeX `\boxed{}`, XML tags `<answer>`, or final sentences).

#### `zelt.is_correct(prediction: str, gold: str) -> bool`
Checks correctness between candidate string and gold reference, normalizing whitespace, commas, fractions, and decimals.

---

### Class `Zelt` (alias: `EarlyStop`)

```python
class Zelt(
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    backend: str = "local",
    base_url: str | None = None,
    decider_path: str | None = None,
    alpha: float = 0.02,
    delta: float = 0.10,
    chunk: int = 48,
    verbose: bool = True,
    lazy: bool = True
)
```

#### Parameters:
- `model`: Model name or HuggingFace ID.
- `backend`: Inference backend (`"local"` for PyTorch, `"ollama"`, or `"vllm"`).
- `base_url`: Host URL for server backends (e.g. `http://localhost:11434` for Ollama).
- `decider_path`: Path to a `.joblib` model file. Defaults to bundled pretrained decider.
- `alpha`: Risk bound limit. Maximum allowable accuracy degradation rate (default: `0.02` = 2%).
- `delta`: Failure probability of the Clopper-Pearson bound (default: `0.10` = 90% confidence).
- `chunk`: Number of generation tokens between probes (default: `48`).
- `verbose`: Whether to output progress logs.
- `lazy`: If `True`, model weights are loaded only on the first inference call.

#### Methods:
- `ask(question: str, verbose: bool | None = None) -> EarlyStopResult`
- `stream(question: str) -> Generator[StreamChunk, None, EarlyStopResult]`
- `batch(questions: list[str], verbose: bool = True) -> BatchResult`
- `fit(data, outdir="results", kind="logreg", seed=0) -> Zelt`
- `record(questions, out="results/trajectories.jsonl") -> Zelt`
- `serve(host="127.0.0.1", port=8000)`
- `info() -> str`

---

### Class `EarlyStopResult` (alias: `ZeltResult`)

#### Attributes:
- `answer` *(str)*: Final parsed answer.
- `tokens` *(int)*: Tokens generated.
- `full_est` *(int)*: Estimated tokens if full CoT had completed.
- `tokens_saved` *(int)*: Difference between `full_est` and `tokens`.
- `saved_pct` *(float)*: Percentage of compute saved.
- `stopped` *(bool)*: `True` if early exit fired; `False` if generation ran to completion.
- `elapsed` *(float)*: Total wall-clock time in seconds.
- `probes` *(list[dict])*: Sequence of probe measurements.
- `scores` *(list[float])*: Calibrated probabilities $P(\text{correct})$ per probe.

#### Methods:
- `cost_saved(price_per_1m_tokens=2.0) -> float`: Calculates estimated dollar savings.
- `carbon_saved(g_co2_per_kwh=385.0, watts_per_device=45.0, tokens_per_sec=35.0) -> float`: Calculates grams of CO₂ emissions avoided.
- `summary() -> str`: Formatted ASCII summary.
- `trace() -> str`: Probe-by-probe table showing timeline, confidence, entropy, and candidate answers.
- `to_dict() -> dict`: Serializes all fields to a Python dictionary.
- `save(path: str)`: Exports result to a JSON file.

---

### Class `StreamChunk`

Yielded during `stream()`.
- `text` *(str)*: The newly generated token string.
- `token_count` *(int)*: Current generation length.
- `is_probe` *(bool)*: Whether this event represents a probe checkpoint.
- `confidence` *(float)*: Geometric mean confidence at this probe.
- `score` *(float)*: Calibrated probability $P(\text{correct})$.
- `stopped` *(bool)*: `True` if this chunk triggered the early exit.
- `answer` *(str | None)*: The final extracted answer if stopped.

---

### Class `BatchResult`

- `results` *(list[EarlyStopResult])*: List of individual query outputs.
- `n` *(int)*: Total questions processed.
- `stopped_count` *(int)*: Count of queries stopped early.
- `stop_rate_pct` *(float)*: Early-exit percentage.
- `total_tokens` *(int)*: Total tokens consumed.
- `total_saved_tokens` *(int)*: Total tokens saved.
- `speedup` *(float)*: Effective speedup ratio (e.g. `1.61x`).
- `export_prometheus() -> str`: Returns Prometheus gauge and counter metrics.
- `save(path: str)`: Writes batch telemetry to JSON.

---

## 6. Mathematical Foundations & Risk Control

### Definition of Harm
Let $A_{\text{full}}$ be the correctness of full unstopped reasoning, and $A_{\text{early}}$ be the correctness of early stopping at threshold $\tau$. Harm occurs when early stopping causes an error that would not have happened under full CoT:

$$\text{Harm} = \mathbb{P}(A_{\text{full}} = 1 \land A_{\text{early}} = 0)$$

### Clopper-Pearson Upper Confidence Bound
Given $N_{\text{cal}}$ held-out calibration examples with $k$ observed harmful early exits, the exact upper Clopper-Pearson confidence bound $\text{UCB}_{1-\delta}(k, N_{\text{cal}})$ satisfies:

$$\text{UCB} = B\left(1 - \delta;\, k + 1,\, N_{\text{cal}} - k\right)$$

where $B(p; a, b)$ is the quantile of the Beta distribution.

### Fixed-Sequence Hypothesis Testing
To prevent multiple testing inflation, candidate thresholds $\tau \in \{0.99, 0.95, 0.90, \dots, 0.50\}$ are evaluated in **strictly descending order** (conservative to aggressive). The optimal threshold $\tau^*$ is the smallest threshold such that:

$$\text{UCB}_{1-\delta}(k(\tau), N_{\text{cal}}) \le \alpha$$

If no threshold satisfies this risk constraint, $\tau^* = \text{None}$ and the system defaults to full CoT generation.

---

## 7. Empirical Benchmark Results

Evaluated on **1,319 real DeepSeek-R1-Distill-Qwen-1.5B** GSM8K trajectories:

| Method | Threshold ($\tau$) | Accuracy | $\Delta$ Acc | Avg Tokens | Token Reduction | Net Speedup | Stop Rate | Harm Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full CoT** | — | **78.25%** | baseline | 750.7 | 0.0% | 1.00× | 0.0% | 0.00% |
| Fixed Budget | 1,632 | 76.74% | -1.51% | 507.7 | 32.4% | 1.47× | 15.1% | 1.51% |
| Raw Confidence | 0.79 | 78.25% | +0.00% | 430.6 | 42.6% | 1.74× | 40.2% | 0.00% |
| **Zelt Decider** | **0.50** | **78.25%** | **+0.00%** | **465.9** | **37.9%** | **1.61×** | **37.8%** | **0.00%** |

- **Calibration Error (ECE)**: Reduced from `0.277` (raw confidence) down to **`0.014`** (Zelt decider).
- **Discrimination (AUROC)**: **`0.9976`**.
- **Empirical Harm**: **`0.00%`** on unseen test split.

---

## 8. CLI Reference Manual

```bash
# Launch Web Studio
zelt ui [--host 127.0.0.1] [--port 8000]

# Ask question directly
zelt ask -q "What is 17 * 23?" [--out result.json]

# Stream live tokens
zelt stream -q "Solve 3x + 9 = 24"

# Process batch file (1 question per line)
zelt batch -i questions.txt -o results.json

# Run real DeepSeek-R1 benchmark
zelt benchmark [--alpha 0.02] [--delta 0.1]

# Train decider on custom trajectories
zelt fit --inp my_trajectories.jsonl --outdir custom_results/

# Run side-by-side demo comparison
zelt demo -q "A store has 150 items..."
```

---

## 9. Custom Model Training & Calibration

To train Zelt on another model (e.g., Llama-3-8B-Instruct or Qwen-2.5-7B-Math):

```python
from zelt import Zelt

zt = Zelt(model="Qwen/Qwen2.5-Math-7B")

# Step 1: Record trajectories on sample questions
dataset = [
    ("What is 2+2?", "4"),
    ("Solve x^2 = 16 for x > 0", "4")
]
zt.record(dataset, out="runs_qwen.jsonl")

# Step 2: Fit and calibrate decider
zt.fit("runs_qwen.jsonl", outdir="results_qwen", alpha=0.02)

# Step 3: Run inference with custom decider
res = zt("What is 15 * 15?")
print(res)
```

---

## 10. Observability & Prometheus Integration

Export metrics directly into Prometheus or Datadog collectors:

```python
from zelt import batch

results = batch(["What is 2+2?", "What is 3*5?"])
print(results.export_prometheus())
```

**Metrics Exported:**
- `zelt_queries_total`: Total questions processed.
- `zelt_early_stops_total`: Total early-exits triggered.
- `zelt_tokens_saved_total`: Total tokens saved across all queries.
- `zelt_stop_rate`: Gauge of early-exit frequency (0.0 to 1.0).
- `zelt_token_reduction_ratio`: Overall compute reduction ratio.
