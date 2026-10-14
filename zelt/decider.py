"""
Learned stop-decider with risk-controlled threshold calibration.

Idea
----
At each probe we have a forced answer plus confidence signals. A small classifier
predicts P(this answer is correct). We stop when that probability >= tau. tau is
chosen on a held-out calibration split with a Learn-then-Test style procedure so that

    P( full CoT correct AND early-stop wrong )  <=  alpha      (w.p. >= 1 - delta)

i.e. a bound on the accuracy we can lose by stopping early.

Everything here is pure numpy/sklearn (no torch), so it works on recorded data.
"""
import json, math, os, csv, random
import numpy as np

from .answers import is_correct, same_answer

FEATURES = ["gm", "mean", "minp", "ent", "logt", "gm_s2", "stable", "run",
            "changes", "dgm", "n_probes",
            # v2 enhanced features
            "d2gm", "run_x_gm", "gm_max", "ans_stable_frac",
            "ent_trend", "minp_s2", "gm_accel_pos"]


# ------------------------------------------------------------------ features
def featurize(probes):
    """One feature row per probe, using only information available at that probe.
    v2: enriched with 7 additional features for better calibration and tighter tau.
    """
    X = []
    hist_gm, hist_ent, hist_minp = [], [], []
    prev, run, changes = None, 0, 0
    prev_gm, prev_dgm = None, None
    gm_max = 0.0
    n_ans, n_stable = 0, 0
    for k, p in enumerate(probes):
        ans = p.get("ans")
        gm = float(p["gm"]) if ans is not None else 0.0
        ent = min(float(p.get("ent") or 0.0), 10.0)
        minp = float(p["minp"]) if ans is not None else 0.0
        hist_gm.append(gm); hist_ent.append(ent); hist_minp.append(minp)
        gm_max = max(gm_max, gm)

        if prev is not None and ans is not None and same_answer(ans, prev):
            run += 1; n_stable += 1
        else:
            if prev is not None and ans is not None:
                changes += 1
            run = 0
        if ans is not None:
            n_ans += 1

        dgm = 0.0 if prev_gm is None else gm - prev_gm
        d2gm = 0.0 if (prev_dgm is None or prev_gm is None) else dgm - prev_dgm

        # gm_s2: smoothed mean of last 2
        gm_s2 = sum(hist_gm[-2:]) / len(hist_gm[-2:])
        minp_s2 = sum(hist_minp[-2:]) / len(hist_minp[-2:])

        # entropy trend: is entropy dropping? (positive = converging)
        ent_trend = 0.0
        if len(hist_ent) >= 2:
            ent_trend = hist_ent[-2] - hist_ent[-1]  # positive = falling entropy = good

        # ans_stable_frac: fraction of probes so far where answer is consistent
        ans_stable_frac = float(n_stable) / max(n_ans, 1) if n_ans > 0 else 0.0

        # run_x_gm: interaction – high run AND high confidence = very reliable
        run_x_gm = float(min(run, 10)) * gm

        # gm_accel_pos: 1 if confidence is accelerating upward
        gm_accel_pos = float(d2gm > 0 and dgm > 0)

        X.append([gm, float(p["mean"]), minp, ent, math.log(max(p["t"], 1)),
                  gm_s2, float(run > 0), float(min(run, 10)),
                  float(min(changes, 10)), dgm, float(k + 1),
                  # v2 features
                  d2gm, run_x_gm, gm_max, ans_stable_frac,
                  ent_trend, minp_s2, gm_accel_pos])
        if ans is not None:
            prev = ans
        prev_gm = gm; prev_dgm = dgm
    return np.array(X, dtype=float).reshape(-1, len(FEATURES))


def probe_labels(ex):
    return np.array([is_correct(p.get("ans"), ex["gold"]) for p in ex["probes"]], dtype=int)


def build_xy(examples):
    Xs, ys = [], []
    for ex in examples:
        if ex["probes"]:
            Xs.append(featurize(ex["probes"]))
            ys.append(probe_labels(ex))
    if not Xs:
        raise ValueError("No probes found in the data.")
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------------------------------------------ decider
class StopDecider:
    def __init__(self, kind="logreg", consensus_k=1):
        """consensus_k: require last K consecutive probes all above tau (reduces false positives)."""
        self.kind, self.model, self.tau, self.meta = kind, None, None, {}
        self.consensus_k = consensus_k

    def _make(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        if self.kind == "logreg":
            return Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=0.5, max_iter=5000, solver="lbfgs"))])
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.04, max_iter=400,
                min_samples_leaf=8, l2_regularization=0.1),
            method="isotonic", cv=3)

    def fit(self, examples):
        X, y = build_xy(examples)
        if len(set(y.tolist())) < 2:
            raise ValueError("Training probes contain only one class (all correct or all wrong); "
                             "need more/varied data.")
        self.model = self._make().fit(X, y)
        self.meta["n_train_probes"] = int(len(y))
        self.meta["train_probe_acc"] = float(y.mean())
        self.meta["n_features"] = len(FEATURES)
        return self

    def score(self, probes):
        """P(correct) at every probe in the list (uses history within the list)."""
        X = featurize(probes)
        if len(X) == 0:
            return np.zeros(0)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        import joblib
        joblib.dump(dict(kind=self.kind, model=self.model, tau=self.tau, meta=self.meta,
                         consensus_k=self.consensus_k), path)

    @staticmethod
    def load(path):
        import joblib
        d = joblib.load(path)
        s = StopDecider(d["kind"], consensus_k=d.get("consensus_k", 1))
        s.model, s.tau, s.meta = d["model"], d["tau"], d["meta"]
        return s


def make_stop_fn(decider, tau=None, consensus_k=None):
    """Live stop rule for EarlyStopper.run().
    Stops when the last consensus_k probes all score >= tau AND have a valid answer.
    Consensus > 1 reduces false positives, allows tighter tau, and improves efficiency.
    """
    tau = decider.tau if tau is None else tau
    k = consensus_k if consensus_k is not None else getattr(decider, "consensus_k", 1)

    def stop(probes):
        if tau is None or len(probes) < k:
            return False
        recent = probes[-k:]
        # All recent probes must have a valid answer
        if any(p.get("ans") is None for p in recent):
            return False
        # All recent probes must agree on the same answer
        first_ans = recent[0]["ans"]
        if not all(p["ans"] == first_ans for p in recent):
            return False
        # Score only needs to pass on the latest probe (scores include history)
        scores = decider.score(probes)
        return float(scores[-1]) >= tau
    return stop


# ------------------------------------------------------------------ simulation
def simulate(examples, seqs, tau):
    """Replay a stopping rule: stop at first probe whose score >= tau."""
    res = []
    for ex, sc in zip(examples, seqs):
        pr, hit = ex["probes"], None
        for k, s in enumerate(sc):
            if s >= tau and pr[k].get("ans") is not None:
                hit = k
                break
        if hit is None:
            tok, ans, ovh, stopped = ex["full_tokens"], ex["full_ans"], sum(q["ptok"] for q in pr), False
        else:
            tok, ans, ovh, stopped = pr[hit]["t"], pr[hit]["ans"], sum(q["ptok"] for q in pr[:hit + 1]), True
        ok = is_correct(ans, ex["gold"])
        res.append(dict(tokens=tok, overhead=ovh, ok=ok, stopped=stopped,
                        harm=bool(ex["full_correct"] and not ok),
                        help=bool((not ex["full_correct"]) and ok)))
    return res


def summarize(res, examples):
    n = len(res)
    full_tok = sum(e["full_tokens"] for e in examples) / n
    full_acc = sum(bool(e["full_correct"]) for e in examples) / n
    tok = sum(r["tokens"] for r in res) / n
    ovh = sum(r["overhead"] for r in res) / n
    acc = sum(r["ok"] for r in res) / n
    return dict(n=n, acc=acc, full_acc=full_acc, acc_delta=acc - full_acc,
                tokens=tok, full_tokens=full_tok,
                reduction_pct=100 * (1 - tok / full_tok),
                reduction_incl_probe_pct=100 * (1 - (tok + ovh) / full_tok),
                stop_rate=sum(r["stopped"] for r in res) / n,
                harm_rate=sum(r["harm"] for r in res) / n,
                help_rate=sum(r["help"] for r in res) / n)


# ------------------------------------------------------------------ risk control
def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    from scipy.stats import beta
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def choose_tau(examples, seqs, grid, alpha, delta):
    """
    Fixed-sequence testing from the most conservative threshold downwards.
    Returns (tau, table). tau=None => no threshold is certifiably safe (never stop).
    """
    n, best, table = len(examples), None, []
    for tau in sorted(grid, reverse=True):
        res = simulate(examples, seqs, tau)
        k = sum(r["harm"] for r in res)
        ucb = cp_upper(k, n, delta)
        table.append(dict(tau=float(tau), harm=k, harm_rate=k / n, ucb=ucb,
                          reduction_pct=100 * (1 - sum(r["tokens"] for r in res) /
                                               sum(e["full_tokens"] for e in examples))))
        if ucb <= alpha:
            best = float(tau)
        else:
            break
    return best, table


# ------------------------------------------------------------------ calibration metrics
def calibration_metrics(y, s, bins=10):
    y, s = np.asarray(y, float), np.asarray(s, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (s >= edges[i]) & ((s < edges[i + 1]) if i < bins - 1 else (s <= edges[i + 1]))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - s[m].mean())
    out = dict(ece=float(ece), brier=float(np.mean((s - y) ** 2)))
    try:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(y, s))
    except Exception:
        out["auroc"] = None
    return out


# ------------------------------------------------------------------ data helpers
def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_examples(rows, seed=0, fracs=(0.5, 0.25, 0.25)):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    a = int(fracs[0] * len(rows)); b = a + int(fracs[1] * len(rows))
    pick = lambda ids: [rows[i] for i in sorted(ids)]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


# ------------------------------------------------------------------ full pipeline
def fit_and_report(rows, outdir, alpha=0.02, delta=0.1, kind="logreg", seed=0, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    rows = [r for r in rows if r["probes"]]
    train, cal, test = split_examples(rows, seed)
    if min(len(train), len(cal), len(test)) < 5:
        raise ValueError("Need more examples (>= ~100 recommended, full GSM8K ideal).")

    dec = StopDecider(kind).fit(train)
    tau_grid = np.unique(np.concatenate([np.arange(0.50, 0.99, 0.01), [0.99, 0.995, 0.999]]))
    seq = {
        "learned": lambda exs: [dec.score(e["probes"]).tolist() for e in exs],
        "confidence": lambda exs: [featurize(e["probes"])[:, FEATURES.index("gm_s2")].tolist() for e in exs],
        "fixed_budget": lambda exs: [[p["t"] for p in e["probes"]] for e in exs],
    }
    budgets = np.unique([p["t"] for e in cal for p in e["probes"]])
    grids = {"learned": tau_grid, "confidence": tau_grid, "fixed_budget": budgets}

    report = dict(n_total=len(rows), n_train=len(train), n_cal=len(cal), n_test=len(test),
                  alpha=alpha, delta=delta, kind=kind, methods={}, calibration={}, tables={})
    for m in ["fixed_budget", "confidence", "learned"]:
        tau, table = choose_tau(cal, seq[m](cal), grids[m], alpha, delta)
        report["tables"][m] = table
        test_seq = seq[m](test)
        if tau is None:
            res = simulate(test, test_seq, float("inf"))
        else:
            res = simulate(test, test_seq, tau)
        s = summarize(res, test); s["chosen_threshold"] = tau
        report["methods"][m] = s
        if m == "learned":
            dec.tau = tau
    # naive (uncalibrated) learned stopper at a fixed 0.9, for contrast
    s = summarize(simulate(test, seq["learned"](test), 0.9), test); s["chosen_threshold"] = 0.9
    report["methods"]["learned_naive_0.9"] = s
    report["methods"]["full_cot"] = dict(
        acc=s["full_acc"], tokens=s["full_tokens"], reduction_pct=0.0, acc_delta=0.0,
        stop_rate=0.0, harm_rate=0.0, n=len(test))

    # calibration quality of the confidence signals on held-out probes
    yt = np.concatenate([probe_labels(e) for e in test])
    st_learned = np.concatenate([dec.score(e["probes"]) for e in test])
    st_gm = np.concatenate([featurize(e["probes"])[:, 0] for e in test])
    report["calibration"] = dict(learned=calibration_metrics(yt, st_learned),
                                 raw_geo_mean_conf=calibration_metrics(yt, st_gm))

    dec.meta.update(alpha=alpha, delta=delta, n_cal=len(cal))
    dec.save(os.path.join(outdir, "decider.joblib"))
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    cols = ["method", "chosen_threshold", "acc", "acc_delta", "tokens", "reduction_pct",
            "reduction_incl_probe_pct", "stop_rate", "harm_rate"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for m, s in report["methods"].items():
            w.writerow([m] + [s.get(c, "") for c in cols[1:]])
    _plots(report, test, dec, outdir)
    if verbose:
        print_report(report)
    return report


def print_report(r):
    print(f"\nExamples: train={r['n_train']} cal={r['n_cal']} test={r['n_test']}   "
          f"risk bound: harm <= {r['alpha']} (conf {1 - r['delta']:.0%})")
    print(f"{'method':<20}{'tau/B':>8}{'acc':>8}{'dAcc':>8}{'tokens':>9}{'reduce%':>9}{'w/probe%':>10}{'stop%':>7}{'harm%':>7}")
    for m in ["full_cot", "fixed_budget", "confidence", "learned", "learned_naive_0.9"]:
        s = r["methods"][m]; t = s.get("chosen_threshold")
        print(f"{m:<20}{('-' if t is None else f'{t:.3g}'):>8}{s['acc']:>8.3f}{s['acc_delta']:>+8.3f}"
              f"{s['tokens']:>9.1f}{s['reduction_pct']:>9.1f}{s.get('reduction_incl_probe_pct', 0):>10.1f}"
              f"{100 * s['stop_rate']:>7.1f}{100 * s['harm_rate']:>7.1f}")
    c = r["calibration"]
    print("\nCalibration on held-out probes (lower ECE/Brier, higher AUROC = better):")
    for k, v in c.items():
        print(f"  {k:<20} ECE={v['ece']:.3f}  Brier={v['brier']:.3f}  AUROC={v['auroc']}")


def _plots(report, test, dec, outdir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    full = report["methods"]["full_cot"]
    for m, mk, col in [("learned", "o", "C0"), ("confidence", "s", "C1")]:
        xs, ys = [], []
        for tau in np.unique(np.concatenate([np.arange(0.5, 0.99, 0.02), [0.99, 0.999]])):
            if m == "learned":
                sq = [dec.score(e["probes"]).tolist() for e in test]
            else:
                sq = [featurize(e["probes"])[:, FEATURES.index("gm_s2")].tolist() for e in test]
            s = summarize(simulate(test, sq, tau), test); xs.append(s["tokens"]); ys.append(s["acc"])
        ax[0].plot(xs, ys, marker=mk, color=col, label=m)
    bs = np.unique([p["t"] for e in test for p in e["probes"]])
    pts = []
    for B in bs[:: max(1, len(bs) // 15)]:
        sq = [[p["t"] for p in e["probes"]] for e in test]
        s = summarize(simulate(test, sq, B), test); pts.append((s["tokens"], s["acc"]))
    if pts:
        ax[0].plot(*zip(*sorted(pts)), "^--", color="C2", label="fixed budget")
    ax[0].scatter([full["tokens"]], [full["acc"]], c="k", marker="*", s=160, label="full CoT", zorder=5)
    ax[0].set_xlabel("avg reasoning tokens"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Accuracy vs tokens (test split)"); ax[0].grid(alpha=.3); ax[0].legend()
    y = np.concatenate([probe_labels(e) for e in test])
    for name, sc, col in [("learned", np.concatenate([dec.score(e["probes"]) for e in test]), "C0"),
                          ("raw confidence", np.concatenate([featurize(e["probes"])[:, 0] for e in test]), "C1")]:
        b = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            m = (sc >= b[i]) & (sc <= b[i + 1] if i == 9 else sc < b[i + 1])
            if m.sum() >= 5:
                xs.append(sc[m].mean()); ys.append(y[m].mean())
        ax[1].plot(xs, ys, "o-", color=col, label=name)
    ax[1].plot([0, 1], [0, 1], "k--", alpha=.5)
    ax[1].set_xlabel("predicted P(correct)"); ax[1].set_ylabel("observed accuracy")
    ax[1].set_title("Reliability diagram (probe answers)"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "report.png"), dpi=150)
    plt.close(fig)
