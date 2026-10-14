"""
High-level Zelt / EarlyStop API — designed for zero-friction use.

    from zelt import Zelt, ask

    # One-liner
    answer = ask("What is 17 * 23?")

    # Object API with streaming, batching, and cost tracking
    zt = Zelt()
    for chunk in zt.stream("What is 17 * 23?"):
        print(chunk.text, end="", flush=True)

That's it. 100% local, no API keys, mathematically risk-controlled.
"""
import os
import json
import time
import math
from typing import Generator, List, Union

_BUNDLED_DECIDER = os.path.join(os.path.dirname(__file__), "data", "pretrained_decider.joblib")
_BUNDLED_DATA    = os.path.join(os.path.dirname(__file__), "data", "gsm8k_real_deepseek_r1.jsonl")
_DEFAULT_MODEL   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ── ANSI colors ─────────────────────────────────────────────────────────────
_RESET = "\033[0m"; _BOLD = "\033[1m"; _GREEN = "\033[92m"
_CYAN  = "\033[96m"; _YELLOW = "\033[93m"; _DIM = "\033[2m"

def _c(text, *codes): return "".join(codes) + str(text) + _RESET


class StreamChunk:
    """
    Event yielded during zt.stream(question).
    """
    def __init__(
        self,
        text: str = "",
        token_count: int = 0,
        is_probe: bool = False,
        probe: dict | None = None,
        score: float | None = None,
        stopped: bool = False,
        answer: str | None = None,
    ):
        self.text        = text
        self.token_count = token_count
        self.is_probe    = is_probe
        self.probe       = probe
        self.score       = score
        self.confidence  = probe.get("gm", 0.0) if probe else (score or 0.0)
        self.stopped     = stopped
        self.answer      = answer

    def __str__(self):
        return self.text

    def __repr__(self):
        if self.stopped:
            return f"StreamChunk(stopped=True, answer={self.answer!r})"
        if self.is_probe:
            return f"StreamChunk(probe, conf={self.confidence:.2f})"
        return f"StreamChunk({self.text!r})"


class EarlyStopResult:
    """
    Return value from Zelt.ask(). Behaves like a string (the final answer),
    but carries full metadata, cost calculations, carbon tracking, and probe traces.
    """
    def __init__(self, answer, tokens, full_est, stopped, probes, scores, elapsed, question=None):
        self.question     = question
        self.answer       = answer
        self.tokens       = tokens
        self.full_est     = full_est          # estimated full-CoT token count
        self.stopped      = stopped           # True if early exit fired
        self.probes       = probes
        self.scores       = scores            # P(correct) per probe
        self.elapsed      = elapsed           # wall-clock seconds
        self.saved_pct    = 100 * (1 - tokens / max(full_est, 1)) if stopped else 0.0
        self.tokens_saved = max(full_est - tokens, 0) if stopped else 0

    def cost_saved(self, price_per_1m_tokens: float = 2.0) -> float:
        """Estimated dollars saved based on output token pricing."""
        return (self.tokens_saved / 1_000_000.0) * price_per_1m_tokens

    def carbon_saved(self, g_co2_per_kwh: float = 385.0, watts_per_device: float = 45.0, tokens_per_sec: float = 35.0) -> float:
        """Estimated grams of CO2 emissions saved by skipping reasoning tokens."""
        if self.tokens_saved <= 0 or tokens_per_sec <= 0:
            return 0.0
        seconds_saved = self.tokens_saved / tokens_per_sec
        kwh_saved = (watts_per_device * (seconds_saved / 3600.0)) / 1000.0
        return kwh_saved * g_co2_per_kwh

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "tokens": self.tokens,
            "full_estimated_tokens": self.full_est,
            "tokens_saved": self.tokens_saved,
            "saved_percent": round(self.saved_pct, 2),
            "stopped_early": self.stopped,
            "elapsed_seconds": round(self.elapsed, 3),
            "cost_saved_usd": round(self.cost_saved(), 6),
            "carbon_saved_grams": round(self.carbon_saved(), 4),
            "probes": self.probes,
            "scores": self.scores,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    # Makes `print(result)` and `str(result)` return the answer
    def __str__(self):  return str(self.answer)
    def __repr__(self): return f"ZeltResult(answer={self.answer!r}, saved={self.saved_pct:.1f}%)"

    def summary(self) -> str:
        lines = [
            _c("─" * 52, _DIM),
            f"  {_c('Answer', _BOLD, _GREEN)}   : {_c(self.answer, _BOLD)}",
            f"  {_c('Tokens', _BOLD)}   : {self.tokens}"
            + (f"  {_c(f'(saved {self.saved_pct:.1f}%)', _GREEN)}" if self.stopped else "  (full CoT)"),
            f"  {_c('Stopped', _BOLD)}  : {_c('YES ⚡', _GREEN) if self.stopped else _c('NO (ran full)', _DIM)}",
            f"  {_c('Time', _BOLD)}     : {self.elapsed:.2f}s",
            f"  {_c('Cost Saved', _BOLD)}: ${_c(f'{self.cost_saved():.5f}', _GREEN)}" if self.stopped else f"  {_c('Cost Saved', _BOLD)}: $0.00",
        ]
        if self.probes:
            last = self.probes[-1]
            score = self.scores[-1] if self.scores is not None and len(self.scores) else 0.0
            lines.append(f"  {_c('Confidence', _BOLD)}: gm={last['gm']:.3f}  P(correct)={score:.3f}")
        lines.append(_c("─" * 52, _DIM))
        return "\n".join(lines)

    def trace(self) -> str:
        """Full probe-by-probe confidence trace."""
        rows = [_c("  t    | P(correct) | gm    | ent  | answer", _DIM)]
        rows.append(_c("  " + "─" * 46, _DIM))
        for p, s in zip(self.probes, self.scores if self.scores is not None else []):
            marker = _c(" ← STOP", _GREEN) if p is self.probes[-1] and self.stopped else ""
            rows.append(f"  {p['t']:4d} | {s:.3f}      | {p['gm']:.3f} | {p.get('ent', 0):.2f} | {p.get('ans')}{marker}")
        return "\n".join(rows)


class BatchResult:
    """
    Collection of EarlyStopResult instances with aggregate statistics and telemetry.
    """
    def __init__(self, results: List[EarlyStopResult], total_time: float):
        self.results            = results
        self.total_time         = total_time
        self.n                  = len(results)
        self.stopped_count      = sum(1 for r in results if r.stopped)
        self.stop_rate_pct      = (100.0 * self.stopped_count / self.n) if self.n else 0.0
        self.total_tokens       = sum(r.tokens for r in results)
        self.total_full_est     = sum(r.full_est for r in results)
        self.total_saved_tokens = sum(r.tokens_saved for r in results)
        self.saved_pct          = (100.0 * self.total_saved_tokens / max(self.total_full_est, 1)) if self.total_full_est else 0.0
        self.speedup            = (self.total_full_est / max(self.total_tokens, 1)) if self.total_tokens else 1.0

    def cost_saved(self, price_per_1m_tokens: float = 2.0) -> float:
        return (self.total_saved_tokens / 1_000_000.0) * price_per_1m_tokens

    def carbon_saved(self, g_co2_per_kwh: float = 385.0) -> float:
        return sum(r.carbon_saved(g_co2_per_kwh=g_co2_per_kwh) for r in self.results)

    def summary(self) -> str:
        lines = [
            _c("═" * 58, _DIM),
            f"  {_c('Zelt Batch Summary', _BOLD, _CYAN)} ({self.n} queries)",
            _c("─" * 58, _DIM),
            f"  {_c('Queries Stopped Early', _BOLD)}: {self.stopped_count}/{self.n} ({self.stop_rate_pct:.1f}%)",
            f"  {_c('Tokens Used', _BOLD)}         : {self.total_tokens:,} (vs {self.total_full_est:,} full CoT)",
            f"  {_c('Tokens Saved', _BOLD)}        : {_c(f'{self.total_saved_tokens:,} ({self.saved_pct:.1f}%)', _GREEN)}",
            f"  {_c('Inference Speedup', _BOLD)}   : {_c(f'{self.speedup:.2f}x faster', _GREEN)}",
            f"  {_c('Est. Cost Saved', _BOLD)}     : {_c(f'${self.cost_saved():.4f}', _GREEN)}",
            f"  {_c('Est. CO₂ Avoided', _BOLD)}    : {self.carbon_saved():.2f} g CO₂",
            f"  {_c('Total Time', _BOLD)}          : {self.total_time:.2f}s",
            _c("═" * 58, _DIM),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "queries": self.n,
            "stopped_early": self.stopped_count,
            "stop_rate_percent": round(self.stop_rate_pct, 2),
            "tokens_used": self.total_tokens,
            "full_cot_tokens": self.total_full_est,
            "tokens_saved": self.total_saved_tokens,
            "saved_percent": round(self.saved_pct, 2),
            "speedup": round(self.speedup, 2),
            "cost_saved_usd": round(self.cost_saved(), 6),
            "carbon_saved_grams": round(self.carbon_saved(), 4),
            "total_time_seconds": round(self.total_time, 2),
            "results": [r.to_dict() for r in self.results]
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def export_prometheus(self) -> str:
        """Export metrics formatted for Prometheus scraping."""
        return (
            f"# HELP zelt_queries_total Total questions processed\n"
            f"# TYPE zelt_queries_total counter\n"
            f"zelt_queries_total {self.n}\n\n"
            f"# HELP zelt_early_stops_total Total early exits triggered\n"
            f"# TYPE zelt_early_stops_total counter\n"
            f"zelt_early_stops_total {self.stopped_count}\n\n"
            f"# HELP zelt_tokens_saved_total Total tokens saved by early stopping\n"
            f"# TYPE zelt_tokens_saved_total counter\n"
            f"zelt_tokens_saved_total {self.total_saved_tokens}\n\n"
            f"# HELP zelt_stop_rate Early exit rate ratio (0 to 1)\n"
            f"# TYPE zelt_stop_rate gauge\n"
            f"zelt_stop_rate {self.stop_rate_pct / 100.0:.4f}\n\n"
            f"# HELP zelt_token_reduction_ratio Compute reduction ratio\n"
            f"# TYPE zelt_token_reduction_ratio gauge\n"
            f"zelt_token_reduction_ratio {self.saved_pct / 100.0:.4f}\n"
        )


class EarlyStop:
    """
    Unified interface for calibrated early-exit reasoning.

    Parameters
    ----------
    model : str
        HuggingFace model ID or server model tag.
    backend : str
        'local' (default PyTorch / HuggingFace), 'ollama', or 'vllm'.
    base_url : str | None
        Server URL when backend is 'ollama' (default http://localhost:11434) or 'vllm' (default http://localhost:8000).
    decider_path : str | None
        Path to a .joblib decider file. None -> uses bundled pretrained decider.
    alpha : float
        Max harm rate bound for calibration (default: 0.02 = 2%).
    delta : float
        Failure probability of the statistical bound (default: 0.10 = 90% confidence).
    chunk : int
        Tokens between probes (default: 48).
    verbose : bool
        Print summary after calls.
    lazy : bool
        If True, delay loading model until first ask/stream call.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        backend: str = "local",
        base_url: str | None = None,
        decider_path: str | None = None,
        alpha: float = 0.02,
        delta: float = 0.10,
        chunk: int = 48,
        verbose: bool = True,
        lazy: bool = True,
    ):
        self._model_name  = model
        self.backend      = backend.lower()
        self.base_url     = base_url
        self._alpha       = alpha
        self._delta       = delta
        self._chunk       = chunk
        self.verbose      = verbose
        self._engine      = None
        self._decider     = None

        path = decider_path or _BUNDLED_DECIDER
        if os.path.exists(path):
            from .decider import StopDecider
            self._decider = StopDecider.load(path)
            if self.verbose:
                tau = self._decider.tau
                print(_c(f"[zelt] Loaded decider  (tau={tau}, harm ≤ {alpha:.0%}, conf {1-delta:.0%})", _DIM))
        else:
            if self.verbose:
                print(_c("[zelt] No pretrained decider found — call .fit() before asking.", _YELLOW))

        if not lazy:
            self._load_engine()

    def _load_engine(self):
        if self._engine is not None:
            return

        if self.backend == "ollama":
            from .adapters import OllamaAdapter
            url = self.base_url or "http://localhost:11434"
            self._engine = OllamaAdapter(model=self._model_name, base_url=url, chunk=self._chunk)
            if self.verbose:
                print(_c(f"[zelt] Connected to Ollama backend at {url}", _GREEN))
        elif self.backend == "vllm":
            from .adapters import VLLMAdapter
            url = self.base_url or "http://localhost:8000"
            self._engine = VLLMAdapter(model=self._model_name, base_url=url, chunk=self._chunk)
            if self.verbose:
                print(_c(f"[zelt] Connected to vLLM backend at {url}", _GREEN))
        else:
            from .core import EarlyStopper
            if self.verbose:
                print(_c(f"[zelt] Loading model: {self._model_name} …", _DIM))
            self._engine = EarlyStopper(self._model_name, chunk=self._chunk)
            if self.verbose:
                dev = self._engine.device
                print(_c(f"[zelt] Model ready on {dev}", _GREEN))

    def ask(self, question: str, verbose: bool | None = None) -> EarlyStopResult:
        """Run early-exit inference on question. Returns an EarlyStopResult."""
        if self._decider is None:
            raise RuntimeError("No decider loaded. Call .fit() or pass decider_path.")
        self._load_engine()

        t0 = time.time()
        from .decider import make_stop_fn
        out = self._engine.run(question, stop_fn=make_stop_fn(self._decider))
        elapsed = time.time() - t0

        full_est = out["tokens"] if not out["stopped"] else int(out["tokens"] / 0.62)
        scores = None
        if out["probes"] and self._decider:
            scores = list(self._decider.score(out["probes"]))

        result = EarlyStopResult(
            question = question,
            answer   = out["answer"],
            tokens   = out["tokens"],
            full_est = full_est,
            stopped  = out["stopped"],
            probes   = out["probes"],
            scores   = scores,
            elapsed  = elapsed,
        )

        show = verbose if verbose is not None else self.verbose
        if show:
            print(result.summary())

        return result

    def reduce_compute_thinking(self, prompt: str, verbose: bool = True) -> EarlyStopResult:
        """
        Primary compute-reduction entrypoint: halts reasoning overthinking
        as soon as confidence is mathematically certified, drastically cutting
        inference time, token generation, and compute energy.

        Parameters
        ----------
        prompt : str
            The input reasoning problem or prompt.
        verbose : bool
            Whether to print compute savings metrics.

        Returns
        -------
        EarlyStopResult:
            Evaluates as the final answer string, carrying full tokens saved,
            speedup, cost, and latency metadata.
        """
        return self.ask(prompt, verbose=verbose)

    def stream(self, question: str) -> Generator[StreamChunk, None, EarlyStopResult]:
        """
        Stream reasoning tokens in real time.
        Yields StreamChunk objects as tokens are generated, with probe pulses and early stop.
        """
        if self._decider is None:
            raise RuntimeError("No decider loaded.")
        self._load_engine()

        from .decider import make_stop_fn
        stop_fn = make_stop_fn(self._decider)
        t0 = time.time()
        probes = []

        if hasattr(self._engine, "run_stream"):
            generator = self._engine.run_stream(question, stop_fn=stop_fn)
            final_res = None
            for ev in generator:
                if ev["type"] == "token":
                    yield StreamChunk(text=ev["text"], token_count=ev["t"])
                elif ev["type"] == "probe":
                    pr = ev["probe"]
                    probes.append(pr)
                    s = self._decider.score([pr])[0] if self._decider else pr.get("gm", 0.0)
                    yield StreamChunk(is_probe=True, probe=pr, score=s, token_count=ev["t"])
                elif ev["type"] == "result":
                    final_res = ev
                    yield StreamChunk(
                        stopped=ev["stopped"],
                        answer=ev["answer"],
                        token_count=ev["tokens"]
                    )
            elapsed = time.time() - t0
            full_est = final_res["tokens"] if not final_res["stopped"] else int(final_res["tokens"] / 0.62)
            scores = list(self._decider.score(probes)) if probes and self._decider else None
            return EarlyStopResult(
                question=question,
                answer=final_res["answer"],
                tokens=final_res["tokens"],
                full_est=full_est,
                stopped=final_res["stopped"],
                probes=probes,
                scores=scores,
                elapsed=elapsed
            )
        else:
            # Fallback for adapters without granular stream
            res = self.ask(question, verbose=False)
            yield StreamChunk(text=res.answer, token_count=res.tokens, stopped=res.stopped, answer=res.answer)
            return res

    def batch(self, questions: List[str], verbose: bool = True) -> BatchResult:
        """
        Process a batch of questions sequentially and return aggregate metrics.
        """
        t0 = time.time()
        results = []
        for i, q in enumerate(questions):
            res = self.ask(q, verbose=False)
            results.append(res)
            if verbose:
                status = "⚡ STOP" if res.stopped else "FULL"
                print(f"  [{i+1}/{len(questions)}] {status} | tokens={res.tokens} (saved {res.saved_pct:.0f}%) | ans: {res.answer}")

        total_time = time.time() - t0
        batch_res = BatchResult(results, total_time)
        if verbose:
            print(batch_res.summary())
        return batch_res

    def serve(self, host: str = "127.0.0.1", port: int = 8000):
        """Launch the interactive web UI dashboard."""
        from .web import run_server
        run_server(self, host=host, port=port)

    def __call__(self, question: str, **kwargs) -> EarlyStopResult:
        """Alias: zt('question') == zt.ask('question')."""
        return self.ask(question, **kwargs)

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(
        self,
        data,
        outdir: str = "results",
        kind: str = "logreg",
        seed: int = 0,
    ) -> "EarlyStop":
        """Train + calibrate decider on recorded trajectories."""
        from .decider import load_rows, fit_and_report, StopDecider

        if isinstance(data, str):
            rows = load_rows(data)
        else:
            rows = list(data)

        print(_c(f"[zelt] Fitting on {len(rows)} examples …", _DIM))
        rep = fit_and_report(rows, outdir, alpha=self._alpha, delta=self._delta,
                             kind=kind, seed=seed, verbose=True)
        self._decider = StopDecider.load(os.path.join(outdir, "decider.joblib"))
        tau = rep["methods"]["learned"]["chosen_threshold"]
        print(_c(f"[zelt] Decider fitted. tau={tau}", _GREEN))
        return self

    def record(
        self,
        questions,
        out: str = "results/trajectories.jsonl",
    ) -> "EarlyStop":
        """Record full CoT trajectories for calibration."""
        from .answers import is_correct
        self._load_engine()
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        done = set()
        if os.path.exists(out):
            with open(out) as f:
                done = {json.loads(l)["idx"] for l in f if l.strip()}
        with open(out, "a") as fo:
            for idx, (q, gold) in enumerate(questions):
                if idx in done:
                    continue
                r = self._engine.run(q)
                row = dict(idx=idx, gold=gold, full_tokens=r["tokens"],
                           full_ans=r["answer"], full_correct=is_correct(r["answer"], gold),
                           probes=r["probes"], hit_max=r["tokens"] >= getattr(self._engine, "max_new", 2048))
                fo.write(json.dumps(row) + "\n"); fo.flush()
                print(f"  [{idx+1}/{len(questions)}] tokens={r['tokens']} ok={row['full_correct']}")
        print(_c(f"[zelt] Saved {len(questions)} trajectories to {out}", _GREEN))
        return self

    def info(self) -> str:
        lines = [
            _c("Zelt", _BOLD, _CYAN) + f"  v{_get_version()}",
            f"  backend  : {self.backend}",
            f"  model    : {self._model_name}",
            f"  device   : {self._engine.device if self._engine else '(not loaded yet)'}",
            f"  alpha    : {self._alpha}  delta: {self._delta}",
        ]
        if self._decider:
            m = self._decider.meta
            lines += [
                f"  tau      : {self._decider.tau}",
                f"  trained  : {m.get('n_train_probes', '?')} probes  acc={m.get('train_probe_acc', '?'):.2%}",
                f"  features : {m.get('n_features', '?')}",
            ]
        return "\n".join(lines)

    def __repr__(self):
        loaded = "model loaded" if self._engine else "lazy"
        dec = f"tau={self._decider.tau}" if self._decider else "no decider"
        return f"Zelt(model={self._model_name!r}, backend={self.backend!r}, {dec}, {loaded})"


def _get_version():
    try:
        from importlib.metadata import version
        return version("zelt")
    except Exception:
        return "0.2.0"


# Aliases
Zelt = EarlyStop
ZeltResult = EarlyStopResult
