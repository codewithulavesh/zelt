#!/usr/bin/env python3
"""
Figure generation script for Zelt paper.
Reads: zelt/data/report.json, zelt/data/gsm8k_real_deepseek_r1.jsonl
Writes: paper/figures/report.png  (already present — this re-generates it)
        paper/figures/token_distribution.png

Usage:
    pip install zelt matplotlib numpy
    python paper/generate_figures.py
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "zelt", "data")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "report.json")) as f:
    report = json.load(f)

with open(os.path.join(DATA_DIR, "gsm8k_real_deepseek_r1.jsonl")) as f:
    rows = [json.loads(line) for line in f if line.strip()]

# ── Figure 1: token distribution ──────────────────────────────────────────
tokens = [r["full_tokens"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 3.5))
bins = np.linspace(0, min(max(tokens), 3000), 50)
ax.hist(tokens, bins=bins, edgecolor="white", color="#3a86ff", alpha=0.85)
ax.axvline(
    np.median(tokens),
    color="#ff006e",
    lw=2,
    linestyle="--",
    label=f"Median = {int(np.median(tokens))}",
)
ax.axvline(
    np.mean(tokens),
    color="#fb5607",
    lw=2,
    linestyle=":",
    label=f"Mean   = {np.mean(tokens):.0f}",
)
ax.set_xlabel("Full CoT tokens", fontsize=12)
ax.set_ylabel("Trajectories", fontsize=12)
ax.set_title(
    "Token distribution: GSM8K × DeepSeek-R1-Distill-Qwen-1.5B\n"
    f"(n={len(rows)}, clipped at 3000; {sum(t > 3000 for t in tokens)} above)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "token_distribution.png"), dpi=150)
plt.close(fig)
print("[+] token_distribution.png written")

# ── Figure 2: accuracy–token Pareto (reconstructed from tables) ────────────
# (Also available as report.png from fit_and_report; this gives vector re-creation)
methods_for_bar = ["full_cot", "fixed_budget", "confidence", "learned", "learned_naive_0.9"]
labels = [
    "Full CoT",
    "Fixed budget\n(B=1632)",
    "Raw conf.\n(τ=0.79)",
    "Zelt learned\n(τ*=0.50)",
    "Naive learned\n(τ=0.90)",
]
accs = [100 * report["methods"][m]["acc"] for m in methods_for_bar]
toks = [report["methods"][m]["tokens"] for m in methods_for_bar]
harms = [100 * report["methods"][m]["harm_rate"] for m in methods_for_bar]

x = np.arange(len(methods_for_bar))
colors = ["#264653", "#e9c46a", "#2a9d8f", "#e76f51", "#a8dadc"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].bar(x, accs, color=colors, edgecolor="white", width=0.6)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Test Accuracy")
axes[0].set_ylim(74, 80)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x, toks, color=colors, edgecolor="white", width=0.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=8)
axes[1].set_ylabel("Mean tokens")
axes[1].set_title("Mean Reasoning Tokens")
axes[1].axhline(750.7, color="black", lw=1, linestyle="--", alpha=0.5)
axes[1].grid(axis="y", alpha=0.3)

axes[2].bar(x, harms, color=colors, edgecolor="white", width=0.6)
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels, fontsize=8)
axes[2].set_ylabel("Harm rate (%)")
axes[2].set_title("Harmful Exit Rate")
axes[2].axhline(2.0, color="#e63946", lw=1.5, linestyle="--", label="α=2%")
axes[2].legend(fontsize=9)
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Zelt vs. baselines (test split, n=331, GSM8K × DeepSeek-R1-Distill-Qwen-1.5B)",
    fontsize=11,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "comparison_bars.png"), dpi=150)
plt.close(fig)
print("[+] comparison_bars.png written")
print("[+] All figures done.")
