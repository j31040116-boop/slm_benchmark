"""
organize_results.py — Rebuilds benchmark_results.json with a clean summary section.
Run once after all tasks are complete.
"""

import json
from datetime import datetime
from pathlib import Path

RESULTS_FILE = "benchmark_results.json"

with open(RESULTS_FILE) as f:
    raw = json.load(f)

tasks_meta = {
    "FPB":       {"full_name": "Financial PhraseBank",              "type": "classification", "metric": "accuracy"},
    "FiQA-SA":   {"full_name": "FiQA Sentiment Analysis",           "type": "classification", "metric": "accuracy"},
    "FiNER-ORD": {"full_name": "Financial NER (FiNER-ORD)",         "type": "classification", "metric": "accuracy"},
    "ConvFinQA": {"full_name": "Conversational Financial QA",       "type": "exact_match",    "metric": "exact_match"},
    "FOMC":      {"full_name": "FOMC Communication Stance",         "type": "classification", "metric": "accuracy"},
    "ECTSum":    {"full_name": "Earnings Call Transcript Sentiment", "type": "classification", "metric": "accuracy"},
    "ACL18":     {"full_name": "Twitter Financial News Sentiment",   "type": "classification", "metric": "accuracy"},
}

# ── Build summary ──────────────────────────────────────────────────────────────
summary_tasks = {}
total_samples = 0
for task, d in raw.items():
    m = d.get("metrics", {})
    n = d.get("n_samples", 0)
    total_samples += n
    ps = d.get("per_sample", [])
    avg_elapsed = round(sum(s["elapsed"] for s in ps) / len(ps), 2) if ps else None
    wall_hrs    = round(sum(s["elapsed"] for s in ps) / 6 / 3600, 2) if ps else None

    summary_tasks[task] = {
        "full_name":       tasks_meta[task]["full_name"],
        "task_type":       tasks_meta[task]["type"],
        "primary_metric":  tasks_meta[task]["metric"],
        "n_samples":       n,
        "metrics":         m,
        "avg_s_per_sample": avg_elapsed,
        "wall_time_hrs":   wall_hrs,
    }

organized = {
    "run_info": {
        "model":           "lfm2.5-thinking:latest",
        "date_organized":  datetime.now().strftime("%Y-%m-%d"),
        "total_samples":   total_samples,
        "n_tasks":         len(raw),
        "workers":         6,
        "max_tokens":      1536,
        "num_ctx":         2048,
    },
    "summary": summary_tasks,
    "per_task_detail": {
        task: {
            "status":     d["status"],
            "n_samples":  d.get("n_samples", 0),
            "metrics":    d.get("metrics", {}),
            "per_sample": d.get("per_sample", []),
        }
        for task, d in raw.items()
    },
}

with open(RESULTS_FILE, "w") as f:
    json.dump(organized, f, indent=2)

print("Organized results written to", RESULTS_FILE)
print(f"\n{'Task':<12} {'Samples':>8}  {'Primary Metric':>16}  {'Macro F1':>10}")
print("-" * 54)
for task, s in summary_tasks.items():
    m = s["metrics"]
    primary = m.get("accuracy", m.get("exact_match", "—"))
    f1 = m.get("macro_f1", "—")
    f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
    print(f"{task:<12} {s['n_samples']:>8}  {primary:>16.4f}  {f1_str:>10}")
