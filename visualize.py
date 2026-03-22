"""
visualize.py — Generate benchmark result charts from benchmark_results.json.

Outputs (saved to ./charts/):
  1. line_accuracy_vs_f1.png  — accuracy vs macro-F1 per task (gap highlighted)
  2. heatmap_metrics.png      — heatmap of all metrics across tasks
  3. pred_distribution.png    — per-task ground truth vs prediction breakdown (%)

Run:
    python visualize.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────────
with open("benchmark_results.json") as f:
    data = json.load(f)

detail = data["per_task_detail"] if "per_task_detail" in data else data
tasks  = list(detail.keys())
OUT    = Path("charts")
OUT.mkdir(exist_ok=True)

# Expected valid labels per task (for pred distribution)
VALID_LABELS = {
    "FPB":       {"negative", "neutral", "positive"},
    "FiQA-SA":   {"negative", "neutral", "positive"},
    "FiNER-ORD": {"entity", "o"},
    "ConvFinQA": None,   # numeric — skip distribution chart
    "FOMC":      {"dovish", "hawkish", "neutral"},
    "FinSent":   {"negative", "neutral", "positive"},
    "ACL18":     {"bearish", "bullish", "neutral"},
}

# Tasks where primary metric is exact_match, not accuracy
EXACT_MATCH_TASKS = {"ConvFinQA"}

sns.set_theme(style="whitegrid", font_scale=1.1)
C_ACC  = "#4C72B0"
C_F1   = "#DD8452"
C_TRUE = "#4C72B0"
C_PRED = "#DD8452"


# ── 1. LINE GRAPH — Accuracy / EM vs Macro F1 ─────────────────────────────────
primaries, f1s = [], []
for t in tasks:
    m = detail[t]["metrics"]
    primaries.append(m.get("accuracy", m.get("exact_match", 0)))
    f1s.append(m.get("macro_f1", None))

x = np.arange(len(tasks))

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(x, primaries, marker="o", linewidth=2.5, markersize=9,
        color=C_ACC, label="Accuracy / Exact-Match", zorder=3)
ax.plot(x, [v if v is not None else np.nan for v in f1s],
        marker="s", linewidth=2.5, markersize=9, linestyle="--",
        color=C_F1, label="Macro F1 (classification tasks only)", zorder=3)

# Shade accuracy–F1 gap
f1_arr  = np.array([v if v is not None else primaries[i] for i, v in enumerate(f1s)])
acc_arr = np.array(primaries)
ax.fill_between(x, f1_arr, acc_arr, alpha=0.12, color="red",
                label="Accuracy–F1 gap (misleading region)")

# Annotate values
for i, (acc, f1) in enumerate(zip(primaries, f1s)):
    ax.annotate(f"{acc:.3f}", (x[i], acc), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=C_ACC)
    if f1 is not None:
        ax.annotate(f"{f1:.3f}", (x[i], f1), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color=C_F1)
    # Mark exact-match tasks
    if tasks[i] in EXACT_MATCH_TASKS:
        ax.annotate("EM", (x[i], acc), textcoords="offset points",
                    xytext=(12, 0), ha="left", fontsize=8, color="grey",
                    style="italic")

ax.axhline(0.333, color="red", linestyle=":", linewidth=1.2, alpha=0.6,
           label="Random baseline (3-class = 0.333)")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "LFM2.5-Thinking — Accuracy vs Macro F1 Across Tasks\n"
    "Shaded gap shows where accuracy overstates real performance  ·  "
    "ConvFinQA uses Exact Match (EM), no F1",
    fontsize=12, fontweight="bold", pad=14,
)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "line_accuracy_vs_f1.png", dpi=150)
plt.close(fig)
print("Saved: charts/line_accuracy_vs_f1.png")


# ── 2. HEATMAP — all metrics ──────────────────────────────────────────────────
metric_keys   = ["accuracy", "macro_f1", "exact_match"]
metric_labels = ["Accuracy", "Macro F1", "Exact Match"]

matrix = np.full((len(metric_keys), len(tasks)), np.nan)
for j, t in enumerate(tasks):
    m = detail[t]["metrics"]
    for i, k in enumerate(metric_keys):
        if k in m:
            matrix[i, j] = m[k]

fig, ax = plt.subplots(figsize=(13, 4))
mask = np.isnan(matrix)
sns.heatmap(
    matrix, annot=True, fmt=".3f", mask=mask,
    xticklabels=tasks, yticklabels=metric_labels,
    cmap="YlOrRd", vmin=0, vmax=1,
    linewidths=0.6, linecolor="white",
    ax=ax, cbar_kws={"label": "Score", "shrink": 0.8},
    annot_kws={"size": 12, "weight": "bold"},
)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if mask[i, j]:
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, fill=True, color="#e8e8e8", zorder=2))
            ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                    fontsize=10, color="#999999", zorder=3)

ax.set_title("LFM2.5-Thinking — Metric Heatmap Across All Tasks",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticklabels(tasks, rotation=0, fontsize=11)
ax.set_yticklabels(metric_labels, rotation=0, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "heatmap_metrics.png", dpi=150)
plt.close(fig)
print("Saved: charts/heatmap_metrics.png")


# ── 3. PRED DISTRIBUTION — percentages so tasks with different n are comparable
dist_tasks = [t for t in tasks if VALID_LABELS.get(t) is not None]
n_plots    = len(dist_tasks)
fig, axes  = plt.subplots(2, 3, figsize=(16, 9))
axes       = axes.flatten()

for idx, t in enumerate(dist_tasks):
    ax    = axes[idx]
    ps    = detail[t].get("per_sample", [])
    valid = VALID_LABELS[t]
    n     = len(ps)
    labels = sorted(valid)

    truth_raw = Counter(s["truth"] for s in ps)
    pred_raw  = Counter(s["pred"]  for s in ps)

    # Convert to percentages
    truth_pct = {l: truth_raw.get(l, 0) / n * 100 for l in labels}
    pred_pct  = {l: pred_raw.get(l, 0)  / n * 100 for l in labels}
    invalid_pct = sum(v for k, v in pred_raw.items() if k not in valid) / n * 100

    if invalid_pct > 0:
        labels_plot = labels + ["invalid"]
        truth_pct["invalid"] = 0.0
        pred_pct["invalid"]  = invalid_pct
    else:
        labels_plot = labels

    xi    = np.arange(len(labels_plot))
    width = 0.38
    t_vals = [truth_pct[l] for l in labels_plot]
    p_vals = [pred_pct[l]  for l in labels_plot]

    b1 = ax.bar(xi - width/2, t_vals, width, label="Ground truth",
                color=C_TRUE, alpha=0.82, edgecolor="white")
    b2 = ax.bar(xi + width/2, p_vals, width, label="Model pred",
                color=C_PRED, alpha=0.82, edgecolor="white")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + max(t_vals + p_vals) * 0.01,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xi)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_title(f"{t}  (n={n:,})", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of samples", fontsize=9)
    ax.set_ylim(0, max(t_vals + p_vals) * 1.18)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "LFM2.5-Thinking — Ground Truth vs Model Predictions by Task (%)\n"
    "ConvFinQA excluded (numeric task)  ·  'invalid' = unparseable model output",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(OUT / "pred_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: charts/pred_distribution.png")


# ── Console summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  LFM2.5-THINKING  BENCHMARK SUMMARY")
print("=" * 55)
print(f"  Model  : lfm2.5-thinking:latest")
print(f"  Tasks  : {len(tasks)}    Total samples: {sum(detail[t]['n_samples'] for t in tasks):,}")
print(f"  NOTE   : Macro F1 is the reliable metric.")
print(f"           Accuracy is inflated by class imbalance.")
print()
print(f"  {'Task':<12} {'Samples':>8}  {'Acc/EM':>8}  {'Macro F1':>10}")
print(f"  {'-' * 44}")
for t in tasks:
    m       = detail[t]["metrics"]
    primary = m.get("accuracy", m.get("exact_match", 0))
    f1      = m.get("macro_f1", "—")
    f1_str  = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
    print(f"  {t:<12} {detail[t]['n_samples']:>8}  {primary:>8.4f}  {f1_str:>10}")
print("=" * 55)
