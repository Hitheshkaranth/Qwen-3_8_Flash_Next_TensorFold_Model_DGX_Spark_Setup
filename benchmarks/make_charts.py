"""Render the README charts from the result JSON files in this folder.

usage: python3 benchmarks/make_charts.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Validated default categorical palette (dataviz skill, references/palette.md), slots 1-3
# (validated all-pairs for CVD in light mode). Fixed per model across every chart.
ORNITH, VLLM_FN, TF = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
LABELS = ["Ornith-1.5-35B-A3B\n(vLLM NVFP4)", "Flash-Next\n(vLLM NVFP4)", "Flash-Next\n(TensorFold INT4)"]
COLORS = [ORNITH, VLLM_FN, TF]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def bar_panel(ax, title, values, fmt, unit, labels=LABELS, colors=COLORS):
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", length=0, labelsize=9.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02, fmt.format(v),
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")


def throughput():
    r = load("throughput_result.json")["results"]
    o = r["ornith-1.5-35b-a3b (vLLM NVFP4, max-num-seqs 20)"]
    v = r["qwen3.8-flash-next (vLLM NVFP4, 10 experts, MAX_NUM_SEQS 16)"]
    t = r["qwen3.8-flash-next-tf (this repo, TensorFold INT4-AR, PARALLEL 16)"]
    peak = lambda m: m[f"users_{m['peak_at_users']}"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    bar_panel(axes[0], "Decode speed, 1 user",
              [o["users_1"]["aggregate_tok_s"], v["users_1"]["aggregate_tok_s"], t["users_1"]["aggregate_tok_s"]],
              "{:.1f}", "tokens / s")
    bar_panel(axes[1], "Peak aggregate decode",
              [peak(o)["aggregate_tok_s"], peak(v)["aggregate_tok_s"], peak(t)["aggregate_tok_s"]],
              "{:.1f}", "tokens / s")
    bar_panel(axes[2], "16 users at once, per user",
              [o["users_16"]["per_user_tok_s"], v["users_16"]["per_user_tok_s"], t["users_16"]["per_user_tok_s"]],
              "{:.1f}", "tokens / s")
    fig.suptitle(f"Raw token rate on one DGX Spark (GB10), 512 output tokens per request. "
                 f"Peaks at {o['peak_at_users']} / {v['peak_at_users']} / {t['peak_at_users']} users.",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "throughput_comparison.png"), dpi=150)
    print("wrote throughput_comparison.png")


def reasoning():
    runs = [load(f) for f in ("reasoning_result_ornith.json", "reasoning_result_vllm_flashnext.json",
                              "reasoning_result_tensorfold.json")]
    if not all(runs):
        print("skipping reasoning chart: a reasoning_result_*.json is missing")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    bar_panel(axes[0], "GSM8K accuracy (100)", [r["gsm8k"]["accuracy"] * 100 for r in runs], "{:.0f}%", "% correct")
    bar_panel(axes[1], "MATH-500 accuracy (100, integer answers)",
              [r["math500"]["accuracy"] * 100 for r in runs], "{:.0f}%", "% correct")
    bar_panel(axes[2], "Tokens per MATH-500 answer", [r["math500"]["avg_tokens"] for r in runs],
              "{:,.0f}", "avg tokens (lower = more direct)")
    for ax in axes[:2]:
        ax.set_ylim(0, 105)
    fig.suptitle("Reasoning and problem solving - same prompts, sampling and limits for all three",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reasoning_comparison.png"), dpi=150)
    print("wrote reasoning_comparison.png")


def knowledge():
    o, t = load("knowledge_result_ornith.json"), load("knowledge_result_tensorfold.json")
    if not (o and t):
        print("skipping knowledge chart: need knowledge_result_ornith.json and knowledge_result_tensorfold.json")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bar_panel(ax, "MMLU-Pro accuracy (700-question stratified sample)",
              [o["accuracy"] * 100, t["accuracy"] * 100], "{:.1f}%", "% correct",
              labels=[LABELS[0], LABELS[2]], colors=[ORNITH, TF])
    ax.set_ylim(0, 100)
    fig.suptitle("Knowledge: MMLU-Pro", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "knowledge_comparison.png"), dpi=150)
    print("wrote knowledge_comparison.png")


throughput()
reasoning()
knowledge()
