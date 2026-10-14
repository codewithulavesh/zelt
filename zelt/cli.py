import argparse
import json
import os
import tempfile


def get_bundled_dataset_path(name="gsm8k_real"):
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "gsm8k_real_deepseek_r1.jsonl"),
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "gsm8k_real_deepseek_r1.jsonl"
        ),
        "data/gsm8k_real_deepseek_r1.jsonl",
        "results/gsm8k_full.jsonl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_questions(name, n):
    if name == "gsm8k_real":
        p = get_bundled_dataset_path("gsm8k_real")
        if not p:
            raise SystemExit(
                "Bundled real dataset not found. Run scripts/build_real_dataset.py first."
            )
        items = []
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    row = json.loads(line)
                    items.append((row["idx"], row.get("question", ""), row["gold"]))
        return items if n <= 0 else items[:n]

    try:
        from datasets import load_dataset

        if name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            items = [
                (i, e["question"], e["answer"].split("####")[-1].strip()) for i, e in enumerate(ds)
            ]
        elif name == "math500":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            items = [(i, e["problem"], e["answer"]) for i, e in enumerate(ds)]
        else:
            raise SystemExit(f"unknown dataset {name}")
        return items if n <= 0 else items[:n]
    except Exception as e:
        bundled = get_bundled_dataset_path("gsm8k_real")
        if bundled:
            print(
                f"Warning: could not load from datasets library ({e}); falling back to bundled real data."
            )
            return load_questions("gsm8k_real", n)
        raise SystemExit(f"Error loading dataset {name}: {e}")


def make_engine(a):
    from .core import EarlyStopper

    return EarlyStopper(a.model, chunk=a.chunk, min_tokens=a.chunk, max_new=a.max_new)


# ------------------------------------------------------------------ record
def cmd_record(a):
    from .answers import is_correct

    eng = make_engine(a)
    items = load_questions(a.dataset, a.n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}
    with open(a.out, "a") as fo:
        for k, (idx, q, gold) in enumerate(items):
            if idx in done:
                continue
            r = eng.run(q)
            row = dict(
                idx=idx,
                gold=gold,
                full_tokens=r["tokens"],
                full_ans=r["answer"],
                full_correct=is_correct(r["answer"], gold),
                probes=r["probes"],
                hit_max=r["tokens"] >= a.max_new,
            )
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            print(
                f"[{k + 1}/{len(items)}] tokens={r['tokens']} probes={len(r['probes'])} "
                f"ok={row['full_correct']}",
                flush=True,
            )


# ------------------------------------------------------------------ fit & benchmark
def _resolve_input(path):
    if os.path.exists(path):
        return path
    bundled = get_bundled_dataset_path()
    if bundled and os.path.exists(bundled):
        print(f"Note: '{path}' not found, using bundled real dataset: {bundled}")
        return bundled
    raise SystemExit(f"Input file '{path}' not found and no bundled dataset available.")


def cmd_fit(a):
    from .decider import fit_and_report, load_rows

    inp = _resolve_input(a.inp)
    fit_and_report(load_rows(inp), a.outdir, alpha=a.alpha, delta=a.delta, kind=a.kind, seed=a.seed)
    print(f"\nSaved: {a.outdir}/decider.joblib, report.json, summary.csv, report.png")


def cmd_benchmark(a):
    """Run comprehensive compute reduction & accuracy testing on real LLM CoT traces."""
    from .decider import fit_and_report, load_rows

    inp = _resolve_input(a.inp)
    rows = load_rows(inp)
    print(f"\nEvaluating LLM compute reduction on {len(rows)} real trajectories ({inp})...")
    rep = fit_and_report(
        rows, a.outdir, alpha=a.alpha, delta=a.delta, kind=a.kind, seed=a.seed, verbose=True
    )

    full = rep["methods"]["full_cot"]
    learned = rep["methods"]["learned"]
    tok_saved = full["tokens"] - learned["tokens"]
    pct_saved = learned["reduction_pct"]
    speedup = full["tokens"] / max(learned["tokens"], 1)

    print("\n" + "=" * 64)
    print("  LLM COMPUTE REDUCTION BENCHMARK REPORT (REAL DATA)")
    print("=" * 64)
    print(
        f"  Base Accuracy (Full CoT)     : {full['acc'] * 100:.2f}% ({full['tokens']:.1f} avg tokens)"
    )
    print(
        f"  Early-Stop Accuracy          : {learned['acc'] * 100:.2f}% ({learned['tokens']:.1f} avg tokens)"
    )
    print(f"  Accuracy Delta               : {learned['acc_delta']:+.2%}")
    print(f"  Reasoning Token Reduction    : {pct_saved:.1f}% ({tok_saved:.1f} tokens/query saved)")
    print(f"  Net Reduction (inc. Probes)  : {learned.get('reduction_incl_probe_pct', 0):.1f}%")
    print(f"  Effective Inference Speedup  : {speedup:.2f}x faster")
    print(
        f"  Early Exit Rate              : {learned['stop_rate'] * 100:.1f}% of queries stopped early"
    )
    print(
        f"  Statistical Risk Bound (α)   : harm <= {rep['alpha']:.2%} (confidence {1 - rep['delta']:.0%})"
    )
    print(f"  Empirical Harm Rate on Test  : {learned['harm_rate'] * 100:.2f}%")
    print("=" * 64 + "\n")


# ------------------------------------------------------------------ live eval
def cmd_live_eval(a):
    """Real early-exit generation on the held-out TEST split; verifies the simulation."""
    from .answers import is_correct
    from .decider import StopDecider, load_rows, split_examples

    dec = StopDecider.load(os.path.join(a.outdir, "decider.joblib"))
    if dec.tau is None:
        raise SystemExit(
            "Decider has no certified threshold (tau=None). Record more data or relax --alpha."
        )
    rows = [r for r in load_rows(a.inp) if r["probes"]]
    _, _, test = split_examples(rows, a.seed)
    test = test[: a.n] if a.n > 0 else test
    qs = {i: (q, g) for i, q, g in load_questions(a.dataset, 0)}
    eng = make_engine(a)
    tot = dict(tok=0, tok_probe=0, ok=0, full_tok=0, full_ok=0, stopped=0)
    for k, ex in enumerate(test):
        q, gold = qs[ex["idx"]]
        r = eng.generate(q, dec)
        ok = is_correct(r["answer"], gold)
        tot["tok"] += r["tokens"]
        tot["tok_probe"] += r["tokens"] + r["probe_tokens"]
        tot["ok"] += ok
        tot["stopped"] += r["stopped"]
        tot["full_tok"] += ex["full_tokens"]
        tot["full_ok"] += bool(ex["full_correct"])
        print(
            f"[{k + 1}/{len(test)}] tokens {r['tokens']} (full {ex['full_tokens']}) ok={ok} "
            f"stopped={r['stopped']}",
            flush=True,
        )
    n = len(test)
    out = dict(
        n=n,
        tau=dec.tau,
        acc=tot["ok"] / n,
        full_acc=tot["full_ok"] / n,
        tokens=tot["tok"] / n,
        full_tokens=tot["full_tok"] / n,
        reduction_pct=100 * (1 - tot["tok"] / tot["full_tok"]),
        reduction_incl_probe_pct=100 * (1 - tot["tok_probe"] / tot["full_tok"]),
        stop_rate=tot["stopped"] / n,
    )
    json.dump(out, open(os.path.join(a.outdir, "live_eval.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------ demo
def cmd_demo(a):
    from .decider import StopDecider

    dec = StopDecider.load(os.path.join(a.outdir, "decider.joblib"))
    eng = make_engine(a)
    q = a.question
    full = eng.run(q)
    early = eng.generate(q, dec)
    print(f"\nFULL  : answer={full['answer']}  tokens={full['tokens']}")
    print(
        f"EARLY : answer={early['answer']}  tokens={early['tokens']}  stopped={early['stopped']}  "
        f"(+{early['probe_tokens']} probe tokens)"
    )
    if early["stopped"]:
        print(f"Saved {100 * (1 - early['tokens'] / max(full['tokens'], 1)):.1f}% reasoning tokens")
    print("\nConfidence trace (t, forced answer, geo-mean conf, P(correct)):")
    scores = dec.score(early["probes"])
    for p, s in zip(early["probes"], scores):
        print(f"  t={p['t']:>5}  ans={p['ans']!s:<10} gm={p['gm']:.2f}  P={s:.2f}")


# ------------------------------------------------------------------ selftest
def cmd_selftest(a):
    from .decider import fit_and_report
    from .synthetic import make_synthetic

    print("SELF-TEST on synthetic data (validates the pipeline, NOT real results)")
    rows = make_synthetic(a.n, seed=1)
    out = a.outdir or tempfile.mkdtemp(prefix="earlystop_selftest_")
    fit_and_report(rows, out, alpha=a.alpha, delta=a.delta, kind=a.kind, seed=0)
    print("\nOutputs in", out)


def cmd_ui(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    zt.serve(host=a.host, port=a.port)


def cmd_stream(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    print(f"\n[zelt] Streaming: {a.question}\n")
    for chunk in zt.stream(a.question):
        if chunk.is_probe:
            print(f" [Probe gm={chunk.confidence:.2f}]", end="", flush=True)
        elif chunk.stopped:
            print(
                f"\n\n⚡ Early exit triggered at token {chunk.token_count}! Answer: {chunk.answer}"
            )
            break
        else:
            print(chunk.text, end="", flush=True)
    print("\n")


def cmd_batch(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    if not os.path.exists(a.inp):
        raise SystemExit(f"Input file not found: {a.inp}")
    with open(a.inp) as f:
        questions = [line.strip() for line in f if line.strip()]
    res = zt.batch(questions)
    if a.out:
        res.save(a.out)
        print(f"Saved batch results to {a.out}")


def cmd_ask(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    res = zt.ask(a.question)
    if a.out:
        res.save(a.out)


def cmd_compare(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    res = zt.compare(a.question, verbose=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"Saved comparison to {a.out}")


def cmd_eval(a):
    from .api import Zelt

    zt = Zelt(model=a.model, backend=a.backend, base_url=a.base_url)
    rep = zt.evaluate(dataset=a.inp, limit=a.limit, alpha=a.alpha, delta=a.delta, verbose=True)
    if a.out:
        rep.save(a.out)
        print(f"Saved evaluation report to {a.out}")


def cmd_serve(a):
    try:
        import uvicorn

        from .server import app

        print(f"Starting production Zelt REST microservice on http://{a.host}:{a.port} ...")
        uvicorn.run(app, host=a.host, port=a.port)
    except ImportError:
        cmd_ui(a)


def main():
    ap = argparse.ArgumentParser(prog="zelt")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        p.add_argument("--backend", default="local", choices=["local", "ollama", "vllm"])
        p.add_argument("--base_url", default=None, help="Server URL for ollama/vllm")
        p.add_argument("--chunk", type=int, default=48)
        p.add_argument("--max_new", type=int, default=2048)
        p.add_argument(
            "--dataset", default="gsm8k_real", choices=["gsm8k_real", "gsm8k", "math500"]
        )

    # UI / Studio
    u = sp.add_parser("ui", help="launch local interactive web studio & visualizer")
    model_args(u)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8000)
    u.set_defaults(f=cmd_ui)

    # Serve (FastAPI production microservice)
    sv = sp.add_parser(
        "serve", help="launch production REST microservice with Prometheus endpoints"
    )
    model_args(sv)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(f=cmd_serve)

    # Ask
    ak = sp.add_parser("ask", help="run early-exit inference on a question")
    model_args(ak)
    ak.add_argument("--question", "-q", required=True)
    ak.add_argument("--out", default=None, help="Save result JSON to path")
    ak.set_defaults(f=cmd_ask)

    # Compare (Side-by-side)
    cmp_p = sp.add_parser("compare", help="run side-by-side full vs early-exit with metrics")
    model_args(cmp_p)
    cmp_p.add_argument("--question", "-q", required=True)
    cmp_p.add_argument("--out", default=None, help="Save comparison JSON")
    cmp_p.set_defaults(f=cmd_compare)

    # Eval (Benchmark audit)
    ev_p = sp.add_parser("eval", help="evaluate dataset with certified Clopper-Pearson risk bound")
    model_args(ev_p)
    ev_p.add_argument("--inp", "-i", default=None, help="JSONL path or None for bundled data")
    ev_p.add_argument("--limit", "-n", type=int, default=None, help="Max examples to evaluate")
    ev_p.add_argument("--alpha", type=float, default=0.02, help="Max harm bound")
    ev_p.add_argument("--delta", type=float, default=0.10, help="Confidence failure rate")
    ev_p.add_argument("--out", "-o", default=None, help="Save evaluation report to JSON")
    ev_p.set_defaults(f=cmd_eval)

    # Stream
    st = sp.add_parser("stream", help="stream tokens with live early-exit detection")
    model_args(st)
    st.add_argument("--question", "-q", required=True)
    st.set_defaults(f=cmd_stream)

    # Batch
    bt = sp.add_parser("batch", help="process batch of questions from a file")
    model_args(bt)
    bt.add_argument("--inp", "-i", required=True, help="Text file with 1 question per line")
    bt.add_argument("--out", "-o", default="results/batch_results.json")
    bt.set_defaults(f=cmd_batch)

    # Benchmark
    b = sp.add_parser(
        "benchmark", help="evaluate LLM compute reduction and risk guarantees on real data"
    )
    b.add_argument("--inp", default="results/gsm8k_full.jsonl")
    b.add_argument("--outdir", default="results")
    b.add_argument("--alpha", type=float, default=0.02, help="max allowed accuracy-loss rate")
    b.add_argument("--delta", type=float, default=0.1, help="failure probability of the guarantee")
    b.add_argument("--kind", default="logreg", choices=["logreg", "gbdt"])
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(f=cmd_benchmark)

    # Fit
    f = sp.add_parser("fit", help="train decider, calibrate threshold, evaluate, plot")
    f.add_argument("--inp", default="results/gsm8k_full.jsonl")
    f.add_argument("--outdir", default="results")
    f.add_argument("--alpha", type=float, default=0.02, help="max allowed accuracy-loss rate")
    f.add_argument("--delta", type=float, default=0.1, help="failure probability of the guarantee")
    f.add_argument("--kind", default="logreg", choices=["logreg", "gbdt"])
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(f=cmd_fit)

    # Record
    r = sp.add_parser("record", help="run full CoT + probes and save to jsonl (resumable)")
    model_args(r)
    r.add_argument("--n", type=int, default=0, help="0 = full test set")
    r.add_argument("--out", default="results/gsm8k_full.jsonl")
    r.set_defaults(f=cmd_record)

    # Live-eval
    p_live = sp.add_parser("live-eval", help="real early-exit generation on the test split")
    model_args(p_live)
    p_live.add_argument("--inp", default="results/gsm8k_full.jsonl")
    p_live.add_argument("--outdir", default="results")
    p_live.add_argument("--n", type=int, default=50)
    p_live.add_argument("--seed", type=int, default=0)
    p_live.set_defaults(f=cmd_live_eval)

    # Demo
    d = sp.add_parser("demo", help="side-by-side full vs early-exit on one question")
    model_args(d)
    d.add_argument("--question", required=True)
    d.add_argument("--outdir", default="results")
    d.set_defaults(f=cmd_demo)

    # Selftest
    s = sp.add_parser("selftest", help="GPU-free pipeline check on synthetic data")
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--outdir", default=None)
    s.add_argument("--alpha", type=float, default=0.03)
    s.add_argument("--delta", type=float, default=0.1)
    s.add_argument("--kind", default="logreg", choices=["logreg", "gbdt"])
    s.set_defaults(f=cmd_selftest)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
