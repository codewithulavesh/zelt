# Zelt: Calibrated, Risk-Controlled Early Exit for Reasoning Large Language Models

**Anonymous Authors**  
*Technical Report & Academic Preprint*  
*October 2026*  

---

## Abstract

Modern reasoning models, such as DeepSeek-R1 and OpenAI o1/o3, achieve state-of-the-art problem-solving performance by generating extended chains-of-thought (CoT) prior to emitting a final answer. However, this test-time compute scaling suffers from severe inefficiency: models frequently solve problems within the first few hundred tokens, but continue generating hundreds of redundant tokens re-verifying, restating, or second-guessing already correct deductions. Existing early-exit techniques either require fine-tuning internal representations, rely on uncalibrated heuristic stopping rules that degrade accuracy, or lack safety guarantees. 

In this paper, we introduce **Zelt**, a lightweight, non-invasive, and mathematically certified early-exit framework for reasoning LLMs. Zelt executes periodic, lightweight probes directly on the existing KV cache, extracts an 18-dimensional feature representation capturing intermediate confidence, sequence entropy, answer stability, and confidence acceleration, and evaluates a statistical decider calibrated via distribution-free risk control. Using exact one-sided Clopper-Pearson bounds with fixed-sequence hypothesis testing, Zelt guarantees that the probability of harmful early exits—cases where full CoT would have succeeded but early exit fails—is bounded by a user-specified risk parameter $\alpha$ (e.g., $2\%$) with $(1-\delta)$ confidence (e.g., $90\%$). 

Evaluated on 1,319 real DeepSeek-R1-Distill-Qwen-1.5B reasoning trajectories on GSM8K, Zelt reduces reasoning tokens by **37.9%** (saving an average of 284.8 tokens per query) and accelerates inference by **1.61×**, with **0.00% accuracy loss** (78.25% baseline vs. 78.25% with Zelt) and zero empirical harm on held-out test data. Zelt requires no model modifications or training, operates with 3× lower Expected Calibration Error ($\text{ECE} = 0.014$ vs. $0.277$), and is released as an open-source library supporting streaming, batching, and local server backends.

---

## 1. Introduction

The paradigm of **test-time compute scaling** has emerged as a cornerstone in modern reasoning language models (Brown et al., 2024; DeepSeek-AI, 2025; Snell et al., 2024). By allocating additional compute during inference through long chain-of-thought (CoT) generation (Wei et al., 2022), models like DeepSeek-R1, OpenAI o1, and Qwen-2.5-Math achieve dramatic improvements across mathematical, logical, and algorithmic benchmarks.

Despite these breakthroughs, test-time compute scaling is fundamentally isotropic: uniform compute budgets are applied across queries regardless of problem complexity. Empirical inspection reveals a ubiquitous phenomenon known as **reasoning overthinking** (Figure 1). For straightforward and moderately challenging tasks, models frequently reach the correct solution within 100–300 tokens. Nevertheless, because the autoregressive decoding process lacks a native mechanism to detect its own convergence, the model continues generating hundreds of additional tokens—often repeating steps, calculating equivalent expressions in alternative notations, or vacillating through speculative verification loops—before finally outputting the concluding answer token.

```
Query: "Natalia sold clips to 48 friends and 24 family members. How many clips did she sell?" (Gold: 72)

❌ Standard Full Chain-of-Thought (782 tokens):
  t=48  : "Natalia sold clips to friends... 48 friends, 24 family..."
  t=96  : "Total clips = 48 + 24 = 72."  <--- Correct answer reached
  t=144 : "Let me verify: 48 + 24. 40 + 20 = 60. 8 + 4 = 12. 60 + 12 = 72."
  t=192 : "Could there be any other interpretation? No, clips sold to friends and family."
  t=240 : "Let's re-read the problem carefully. 48 clips to friends... 24 clips to family..."
  ... [542 additional redundant tokens] ...
  t=782 : "</think> Natalia sold 72 clips."

⚡ With Zelt (112 tokens, 85.7% compute reduction):
  t=48  : Probe answer = None (entropy=2.81, gm=0.22) -> Continue
  t=96  : Probe answer = "72" (entropy=0.62, gm=0.98, P(correct)=0.99) -> STOP EXECUTED
  Result: 72 (Correct). Saved 670 tokens with zero accuracy degradation.
```
*Figure 1: Comparison between standard unstopped reasoning generation and Zelt early-exit.*

Halting generation early without sacrificing reasoning accuracy poses three fundamental technical challenges:

1. **Uncalibrated Model Confidence**: Raw autoregressive token probabilities and sequence geometric means are notoriously overconfident (Guo et al., 2017). An intermediate candidate answer with high nominal probability may still stem from flawed preliminary reasoning.
2. **Computational Overhead of Verification**: Invoking external verifier models (Cobbe et al., 2021) or reward models at every step introduces prohibitive latency and GPU memory overhead, negating any compute savings.
3. **Absence of Rigorous Safety Guarantees**: In production systems, random or heuristic early exits that degrade accuracy by unquantifiable margins are unacceptable. A deployment requires formal statistical guarantees that performance will not deteriorate beyond a certified bound.

### Contributions
In this work, we propose **Zelt**, a calibrated, risk-controlled framework for early exiting in reasoning LLMs:
- **Zero-Overhead Greedy Probing**: We perform lightweight intermediate probes directly into the ongoing Key-Value (KV) cache every $k$ tokens, avoiding auxiliary verifier models and requiring no model weights modification.
- **Multimodal Confidence Featurization**: We design an 18-dimensional feature representation combining instantaneous token log-probabilities, sequence entropy, stability across consecutive probes, run-length interactions, and confidence acceleration (second derivative).
- **Distribution-Free Risk Control via Clopper-Pearson Bounds**: We formulate early-exit safety as a hypothesis testing problem over a held-out calibration split. Using exact one-sided Clopper-Pearson bounds under fixed-sequence testing, Zelt guarantees that the **harm rate**—the probability that early exit causes an error on a question the full model would have solved—is bounded by $\alpha$ with probability at least $1-\delta$.
- **Empirical Validation on Real Trajectories**: On 1,319 real trajectories from DeepSeek-R1-Distill-Qwen-1.5B on GSM8K, Zelt achieves a **37.9% token reduction** (1.61× inference speedup) with **0.00% accuracy loss** and an Expected Calibration Error of **0.014** (down from 0.277).
- **Production-Ready Open-Source Library**: We release an end-to-end Python library featuring a zero-config one-liner API, real-time token streaming, batch processing with cost/carbon accounting, an interactive web studio, and Ollama/vLLM adapters.

---

## 2. Related Work

### 2.1 Test-Time Reasoning and Compute Scaling
Recent advances in Large Language Models (LLMs) have established that scaling inference compute yields significant accuracy gains on complex reasoning benchmarks (Snell et al., 2024; Brown et al., 2024). Techniques such as Chain-of-Thought prompting (Wei et al., 2022), self-consistency sampling (Wang et al., 2022), tree-of-thought search (Yao et al., 2024), and reinforcement-learning-driven reasoning models such as DeepSeek-R1 (DeepSeek-AI, 2025) and OpenAI o1/o3 explicitly rely on extended token budgets. However, prior investigations note that reasoning models exhibit extensive verbosity and overthinking on easy queries (Snell et al., 2024; DeepSeek-AI, 2025). Zelt directly targets this inefficiency by adaptively truncating compute once convergence is certified.

### 2.2 Early Exit in Deep Neural Networks
Early exit architectures have a rich history in computer vision (Teerapittayanon et al., 2016) and standard Transformer language models (Schuster et al., 2022; Elhoushi et al., 2024; Bae et al., 2023). Most existing early exit approaches (e.g., DeeBERT, FastBERT) attach auxiliary classification heads to intermediate hidden layers within the network backbone and exit across the depth dimension. In contrast, Zelt operates across the **temporal / sequence dimension** during autoregressive generation. Crucially, Zelt does not require modifying model architectures, retraining weights, or adding layer heads.

### 2.3 Conformal Prediction and Distribution-Free Risk Control
Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2023) provides a framework for distribution-free uncertainty quantification. Recent works have applied conformal prediction to natural language generation (Quach et al., 2023; Kumar et al., 2023). Learn-Then-Test (Angelopoulos et al., 2021) and distribution-free risk control provide statistical bounds on arbitrary monotonic loss functions. Zelt builds upon these foundations by treating early-exit harm as a Bernoulli risk parameter, using exact Clopper-Pearson confidence bounds combined with fixed-sequence testing to calibrate stopping thresholds without heuristic tuning.

---

## 3. Methodology

Let $x \in \mathcal{X}$ denote an input reasoning prompt, and let $y = (y_1, y_2, \dots, y_T)$ denote the sequence of autoregressively generated reasoning tokens under greedy decoding from model $M$, ending in an extracted answer $\hat{a}_{\text{full}} = \text{Extract}(y)$. Let $a^* \in \mathcal{A}$ denote the ground-truth answer. The full generation accuracy is the indicator variable:

$$A_{\text{full}} = \mathbb{I}(\hat{a}_{\text{full}} = a^*)$$

Our objective is to identify a stopping time $t_{\text{stop}} \le T$ that produces an early answer $\hat{a}_{\text{early}} = \hat{a}_{t_{\text{stop}}}$ with accuracy $A_{\text{early}} = \mathbb{I}(\hat{a}_{\text{early}} = a^*)$, such that the token length $t_{\text{stop}} \ll T$ while maintaining $A_{\text{early}} \approx A_{\text{full}}$.

### 3.1 Lightweight KV-Cache Probing

Rather than attaching auxiliary classification heads or querying an external verifier, Zelt performs non-invasive probing directly on the model's active Key-Value (KV) cache. At regular intervals $t = k, 2k, 3k, \dots$ (with default chunk size $k = 48$ tokens):

1. The ongoing generation state is retained in the KV cache $\mathcal{K}_t, \mathcal{V}_t$.
2. A temporary probe branch generates a short greedy sequence of maximum length $L_{\text{probe}} \approx 10$ tokens to attempt extracting the intermediate candidate answer $\hat{a}_t$.
3. The probe tokens are discarded, leaving the primary reasoning trajectory unmodified.

Because the KV cache is already computed, the probe requires only forward passes on $L_{\text{probe}}$ tokens, adding negligible compute overhead ($< 3\%$ of total FLOPs).

### 3.2 18-Dimensional Confidence Featurization

Raw confidence metrics (e.g. sequence geometric-mean probability) are insufficient for accurate early stopping. Zelt extracts an 18-dimensional feature vector $\phi_t \in \mathbb{R}^{18}$ at each probe $t$:

| Feature Category | Features | Description |
|:---|:---|:---|
| **Instantaneous Confidence** | `gm`, `mean`, `minp` | Geometric mean, arithmetic mean, and minimum token probability of the extracted probe answer. |
| **Information Entropy** | `ent`, `first_ent`, `d_ent` | Shannon entropy $H(P_t) = -\sum p_i \log p_i$ of next-token distribution; capture ambiguity at branching points. |
| **Temporal Dynamics** | `logt`, `n_probes` | $\log(t)$ and count of probes conducted so far; models prior that longer CoT has higher baseline readiness. |
| **Answer Stability** | `stable`, `run`, `changes` | Binary indicator if $\hat{a}_t == \hat{a}_{t-1}$; run length of consecutive identical answers; count of answer flips. |
| **Confidence Momentum** | `dgm`, `d2gm`, `gm_s2` | First derivative $\Delta \text{gm} = \text{gm}_t - \text{gm}_{t-1}$; second derivative (acceleration) $\Delta^2 \text{gm}$; running standard deviation of confidence. |
| **Historical Maximums** | `gm_max`, `run_x_gm` | Maximum confidence observed so far; interaction between run length and confidence. |
| **Global Stability** | `stability_fraction` | Ratio of stable probe transitions over total probes: $\frac{1}{N-1} \sum \mathbb{I}(\hat{a}_i = \hat{a}_{i-1})$. |

### 3.3 Calibrated Statistical Decider

We train a lightweight logistic classifier $f_\theta: \mathbb{R}^{18} \to [0, 1]$ on a training split of recorded trajectories to output the calibrated posterior probability:

$$S_t = f_\theta(\phi_t) = \sigma(\theta^\top \phi_t) \approx \mathbb{P}(\hat{a}_t = a^* \mid \phi_t)$$

The decision rule evaluates whether the score exceeds a threshold $\tau \in [0, 1]$:

$$\text{Decide}(t) = \begin{cases} \text{STOP}, & \text{if } S_t \ge \tau \text{ and } \hat{a}_t \neq \text{None} \\ \text{CONTINUE}, & \text{otherwise} \end{cases}$$

---

## 4. Statistical Risk Control Guarantees

In real-world deployment, practitioners require rigorous guarantees that early stopping will not silently degrade accuracy. We formulate this guarantee using the framework of distribution-free risk control.

### 4.1 Definition of Harm
We define **Harm** as the event where the full unstopped Chain-of-Thought is correct, but the early exit produces an incorrect answer:

$$\text{Harm}(x; \tau) = \mathbb{I}\left( A_{\text{full}}(x) = 1 \;\land\; A_{\text{early}}(x; \tau) = 0 \right)$$

Note that if the full CoT was already incorrect ($A_{\text{full}} = 0$), an incorrect early exit does not constitute harm, as the baseline model itself failed on the query.

### 4.2 Exact Clopper-Pearson Upper Bound
Let $\mathcal{D}_{\text{cal}} = \{x_1, \dots, x_{N_{\text{cal}}}\}$ be an independent held-out calibration dataset exchangeable with the test distribution. For any fixed threshold $\tau$, let:

$$K(\tau) = \sum_{i=1}^{N_{\text{cal}}} \text{Harm}(x_i; \tau)$$

be the number of harmful early exits observed on the calibration set. Because $K(\tau) \sim \text{Binomial}(N_{\text{cal}}, R(\tau))$ where $R(\tau) = \mathbb{E}[\text{Harm}(X; \tau)]$, the exact one-sided Clopper-Pearson upper confidence bound at level $1-\delta$ is given by:

$$\overline{R}_{1-\delta}(\tau) = \text{BetaQuantile}\left(1 - \delta;\; K(\tau) + 1,\; N_{\text{cal}} - K(\tau)\right)$$

By construction, under exchangeability:

$$\mathbb{P}\left( R(\tau) \le \overline{R}_{1-\delta}(\tau) \right) \ge 1 - \delta$$

### 4.3 Multiple Testing via Fixed-Sequence Testing
To choose the most aggressive (lowest) threshold $\tau$ that satisfies the risk constraint $R(\tau) \le \alpha$ without inflating Family-Wise Error Rate (FWER), we discretize candidate thresholds in strictly descending order:

$$\mathcal{T} = \langle \tau_1 = 0.99,\, \tau_2 = 0.95,\, \tau_3 = 0.90,\, \dots,\, \tau_M = 0.50 \rangle$$

We evaluate hypotheses $H_{0, m}: R(\tau_m) > \alpha$ sequentially for $m = 1, 2, \dots, M$. The fixed-sequence testing principle (Maurer et al., 1995) guarantees that testing along a predefined sequence controls FWER at level $\delta$ with **no Bonferroni penalty**:

$$\tau^* = \min \left\{ \tau_m \in \mathcal{T} \;\Big|\; \forall j \le m,\, \overline{R}_{1-\delta}(\tau_j) \le \alpha \right\}$$

If no threshold passes the test, $\tau^* = \text{None}$, and Zelt automatically falls back to full CoT generation, guaranteeing that the system never violates the safety budget.

---

## 5. Experimental Evaluation

### 5.1 Experimental Setup
- **Base Model**: DeepSeek-R1-Distill-Qwen-1.5B (reasoning model fine-tuned with reinforcement learning).
- **Benchmark**: GSM8K (Grade School Math 8K; Cobbe et al., 2021).
- **Data Splits**: 1,319 full trajectories split strictly into Train ($N_{\text{train}} = 659$), Calibration ($N_{\text{cal}} = 329$), and Held-Out Test ($N_{\text{test}} = 331$).
- **Parameters**: Risk parameter $\alpha = 0.02$ (max 2% harm rate), confidence level $1-\delta = 0.90$ (90% confidence), chunk size $k = 48$.

### 5.2 Main Benchmark Results

Table 1 summarizes the performance of Zelt compared to competitive baselines on the held-out test split (331 questions).

| Method | Threshold ($\tau$) | Accuracy | $\Delta$ Acc | Avg Tokens | Token Reduction | Net Speedup | Stop Rate | Harm Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full CoT (baseline)** | — | **78.25%** | baseline | 750.7 | 0.0% | 1.00× | 0.0% | 0.00% |
| Fixed Token Budget | 1,632 | 76.74% | -1.51% | 507.7 | 32.4% | 1.47× | 15.1% | 1.51% |
| Raw Confidence | 0.79 | 78.25% | +0.00% | 430.6 | 42.6% | 1.74× | 40.2% | 0.00% |
| **Zelt Decider (Ours)** | **0.50** | **78.25%** | **+0.00%** | **465.9** | **37.9%** | **1.61×** | **37.8%** | **0.00%** |

*Table 1: Main benchmark results on held-out test split of real DeepSeek-R1 trajectories.*

#### Key Findings:
1. **Zero Accuracy Degradation**: Zelt achieves identical accuracy to full CoT generation (78.25%) while slashing average reasoning length from 750.7 to 465.9 tokens.
2. **Superiority over Fixed Budgets**: Truncating generation at a fixed token limit degrades accuracy by $-1.51\%$ and incurs a 1.51% harm rate. Zelt's adaptive stopping avoids premature truncation on hard queries.
3. **Statistical Safety**: The empirical harm rate of Zelt on unseen test data is **0.00%**, strictly satisfying the theoretical bound of $\alpha \le 2\%$.

### 5.3 Calibration Analysis
To evaluate whether the decider provides meaningful probabilities, we measure Expected Calibration Error (ECE), Brier Score, and Area Under the ROC Curve (AUROC) on held-out probe instances:

| Metric | Raw Geometric-Mean Confidence | Zelt Decider | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Expected Calibration Error (ECE) ↓** | 0.277 | **0.014** | **19.8× Better Calibration** |
| **Brier Score ↓** | 0.195 | **0.004** | **48.7× Lower Error** |
| **AUROC ↑** | 0.9970 | **0.9976** | **High Discrimination** |

*Table 2: Probability calibration and discriminative metrics.*

As shown in Table 2, raw token probabilities suffer from severe miscalibration ($\text{ECE} = 0.277$). Zelt's 18-feature representation achieves near-perfect diagonal calibration ($\text{ECE} = 0.014$), allowing threshold selection to translate directly to reliable risk bounds.

---

## 6. Implementation & System Features

Zelt is designed for drop-in developer usability with zero configuration.

### 6.1 Unified Python API
```python
from zelt import reduce_compute_thinking, ask, stream, batch, Zelt

# 1. Primary compute-reduction entrypoint
result = reduce_compute_thinking("What is 17 * 23?")
print("Answer:", result)  # "391"
print(result.summary())  # Displays tokens saved (37.9%), speedup (1.61x), cost saved

# 2. Real-time streaming generator
for chunk in stream("What is 17 * 23?"):
    if chunk.is_probe:
        print(f" [Probe gm={chunk.confidence:.2f}]", end="")
    elif chunk.stopped:
        print(f"\n⚡ Early exit at token {chunk.token_count}! Answer: {chunk.answer}")
        break
    else:
        print(chunk.text, end="", flush=True)

# 3. Batch inference with dollar & CO2 tracking
results = batch(["What is 2+2?", "Solve 3x + 9 = 24"])
print(results.summary())
print(results.export_prometheus())
```

### 6.2 External Server Adapters (Ollama & vLLM)
Zelt includes native streaming adapters for external inference engines, enabling early stopping without running raw PyTorch weights locally:
```python
# Connect to local Ollama server
zt = Zelt(backend="ollama", model="deepseek-r1:1.5b")

# Connect to local vLLM / TGI OpenAI-compatible server
zt = Zelt(
    backend="vllm",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    base_url="http://localhost:8000",
)
```

### 6.3 Interactive Web Studio
Zelt bundles a zero-dependency web visualizer (`zelt ui`) built using Python's standard `http.server` and Server-Sent Events (SSE). The interface displays:
- Real-time token streaming with visual probe pulses.
- Interactive confidence timeline charts showing $P(\text{correct})$ progression.
- Side-by-side token reduction gauges and dollar/carbon savings calculators.

### 6.4 Integration into Large LLM Systems
In real-world deployments, `reduce_compute_thinking` serves as a drop-in replacement across multi-tiered AI architectures:
1. **Direct HuggingFace Integration via `ZeltStoppingCriteria`**:
   Passing `ZeltStoppingCriteria(tokenizer)` to standard `model.generate(..., stopping_criteria=[...])` dynamically halts autoregressive GPU decoding steps without altering core modeling code.
2. **REST / Microservice Serving**:
   Wrapping `reduce_compute_thinking` inside FastAPI/Triton inference routes eliminates GPU memory saturation by throttling redundant thinking tokens before transmission.
3. **Compound AI Systems & Agents**:
   Agents invoking multi-step sub-routines (e.g. LangChain, CrewAI) bind `reduce_compute_thinking` as their mathematical/logical reasoning tool, preventing single sub-goals from exhausting the collective context window.

---

## 7. Discussion & Limitations

### 7.1 Scope of Probing
Zelt's intermediate probes currently rely on greedy decoding for $L_{\text{probe}} = 10$ tokens. While highly effective for math, symbolic logic, and structured extraction, tasks requiring extended essay responses may require alternative probe extractors or multi-step consistency checks.

### 7.2 Exchangeability Assumption
The statistical guarantee relies on the standard conformal assumption that calibration and test data are independent and identically distributed (or exchangeable). Under severe out-of-distribution shifts (e.g., deploying a decider calibrated on grade-school math onto graduate-level physics proofs), conservative safety thresholds ($\alpha = 0.01$) or continuous recalibration are recommended.

---

## 8. Conclusion

We presented **Zelt**, a calibrated, risk-controlled early-exit framework for reasoning Large Language Models. By combining periodic KV-cache probing, an 18-signal confidence representation, and exact Clopper-Pearson risk bounds with fixed-sequence testing, Zelt eliminates the pervasive overthinking problem in reasoning models without sacrificing accuracy. On real DeepSeek-R1 benchmarks, Zelt delivers a **37.9% compute reduction** (1.61× inference speedup) with **0.00% accuracy loss** and mathematically certified safety guarantees.

---

## References

1. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
2. Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Malik, J. (2021). Learn then test: Calibrating predictive algorithms to higher-order guarantees. *arXiv preprint arXiv:2110.01052*.
3. Bae, S., Ko, J., Song, H., & Yun, S. Y. (2023). Fast and robust early-exiting for large language models. *ICLR 2023*.
4. Brown, B., Juravsky, J., Ryan, C., et al. (2024). Large language monkeys: Scaling inference compute with repeated sampling. *arXiv preprint arXiv:2407.21787*.
5. Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.
6. Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.
7. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint arXiv:2501.12948*.
8. Elhoushi, M., Shrivastava, A., Lojek, D., et al. (2024). LayerSkip: Enabling early exit inference and self-speculative decoding. *arXiv preprint arXiv:2404.16710*.
9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
10. Kumar, V., Malladi, S., & Liang, P. (2023). Conformal risk control for language generation. *NeurIPS 2023*.
11. Maurer, W., Hothorn, L., & Lehmacher, W. (1995). Multiple comparisons in drug clinical trials and preclinical assays: problems and solutions. *Biometrie in der chemisch-pharmazeutischen Industrie*, 6, 89-108.
12. Quach, V., Fisch, A., Schuster, T., et al. (2023). Conformalized language modeling. *ICLR 2024*.
13. Schuster, T., Fisch, A., Gupta, J., et al. (2022). Confident adaptive language modeling. *NeurIPS 2022*.
14. Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*.
15. Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast inference via early exiting from deep neural networks. *ICPR 2016*.
16. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer Science & Business Media.
17. Wang, X., Wei, J., Schuurmans, D., et al. (2022). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.
18. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.
19. Yao, S., Yu, D., Zhao, J., et al. (2024). Tree of thoughts: Deliberate problem solving with large language models. *NeurIPS 2023*.
