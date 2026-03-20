"""
slm_eval.py — Main evaluation harness for LFM2.5-1.2B-Thinking via Ollama.

Usage:
    python slm_eval.py                    # run all tasks in parallel
    python slm_eval.py --tasks FPB ACL18  # run specific tasks
    python slm_eval.py --no-debug         # full dataset (override DEBUG_MODE)
    python slm_eval.py --no-resume        # ignore existing results and re-run
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
from datasets import load_dataset

import config
from config import TASKS, RESULTS_FILE, SAVE_EVERY, DEBUG_MODE
from parsers import extract_think, extract_result, parse_float, normalise_label
from prompts import get_prompt
from metrics import compute_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results() -> dict:
    p = Path(RESULTS_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(results: dict) -> None:
    Path(RESULTS_FILE).write_text(json.dumps(results, indent=2))


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_task_dataset(task_key: str, cfg: dict):
    kwargs = {"trust_remote_code": True}
    if cfg.get("hf_name"):
        kwargs["name"] = cfg["hf_name"]

    gated = cfg.get("gated", False)

    try:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"], **kwargs)
    except Exception as exc:
        if gated:
            log(f"  [WARN] Could not load gated dataset {task_key}: {exc}")
            return []
        raise

    text_col  = cfg["text_col"]
    label_col = cfg["label_col"]
    label_map = cfg.get("label_map", {})
    samples   = []

    for row in ds:
        if task_key == "FiNER-ORD":
            text      = str(row[text_col])
            raw_label = row[label_col]
            label     = "entity" if raw_label != 0 else "O"
        else:
            text      = str(row[text_col])
            raw_label = row[label_col]
            if isinstance(raw_label, int) and label_map:
                label = label_map[raw_label]
            else:
                label = str(raw_label)
        samples.append((text, label))

    return samples


# ── Model call ────────────────────────────────────────────────────────────────

def call_model(prompt: str) -> str:
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": config.MAX_TOKENS},
        think=config.THINK_MODE,
    )
    raw = response["message"]["content"]
    thinking = getattr(response["message"], "thinking", "") or ""
    if thinking:
        raw = f"<think>{thinking}</think>\n{raw}"
    return raw


# ── Per-sample worker (runs in thread pool) ───────────────────────────────────

def process_one(task_key: str, cfg: dict, idx: int, text: str, label: str):
    task_type = cfg["type"]
    prompt    = get_prompt(task_key, text)
    t0        = time.time()
    try:
        raw = call_model(prompt)
    except Exception as exc:
        log(f"  [ERROR] {task_key} sample {idx}: {exc}")
        return None
    elapsed = time.time() - t0
    extract_think(raw, task=task_key, sample_id=idx)
    result = extract_result(raw)

    if task_type == "classification":
        pred  = normalise_label(result)
        truth = normalise_label(str(label))
    elif task_type == "regression":
        pred  = parse_float(result)
        truth = parse_float(str(label))
        if pred is None:
            pred = 0.0
    elif task_type == "exact_match":
        pred  = result.strip()
        truth = str(label).strip()
    else:
        pred  = result
        truth = str(label)

    return {
        "truth":   truth,
        "pred":    pred,
        "elapsed": elapsed,
        "sample":  {
            "id":      idx,
            "text":    text[:200],
            "truth":   truth,
            "pred":    pred,
            "correct": truth == pred if task_type != "regression" else None,
            "elapsed": round(elapsed, 2),
        },
    }


# ── GPU check ─────────────────────────────────────────────────────────────────

def check_gpu() -> None:
    try:
        info   = ollama.ps()
        models = getattr(info, "models", []) or []
        if models:
            for m in models:
                name      = getattr(m, "name", "?")
                size_vram = getattr(m, "size_vram", 0)
                log(f"  [GPU] {name} — VRAM in use: {size_vram / 1e9:.1f} GB")
        else:
            log("  [GPU] No models loaded yet — will load on first call.")
        log("  [GPU] On Apple M4, Ollama uses Metal automatically. You're on GPU.")
    except Exception as exc:
        log(f"  [WARN] Could not reach Ollama: {exc}")
        log("  [WARN] Make sure 'ollama serve' is running in another terminal.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SLM Financial Benchmark Harness")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    parser.add_argument("--no-debug",  action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    log("\n[SYSTEM CHECK]")
    check_gpu()

    debug   = DEBUG_MODE and not args.no_debug
    results = {} if args.no_resume else load_results()

    if debug:
        log("[DEBUG MODE] Only first 3 samples per task.")

    # ── Load all datasets ──────────────────────────────────────────────────
    log("\n[LOADING DATASETS]")
    task_data = {}
    for task_key in args.tasks:
        if task_key not in TASKS:
            log(f"[ERROR] Unknown task '{task_key}'. Valid: {list(TASKS.keys())}")
            sys.exit(1)
        if task_key in results and results[task_key].get("status") == "complete":
            log(f"  {task_key}: already complete — skipping")
            continue
        cfg     = TASKS[task_key]
        samples = load_task_dataset(task_key, cfg)
        if not samples:
            results[task_key] = {"status": "skipped", "reason": "dataset unavailable"}
            save_results(results)
            continue
        if debug:
            samples = samples[:3]
        per_task_cap = cfg.get("max_samples")
        cap = min(c for c in [config.MAX_SAMPLES, per_task_cap] if c is not None) if any(c is not None for c in [config.MAX_SAMPLES, per_task_cap]) else None
        if cap is not None:
            samples = samples[:cap]
        task_data[task_key] = samples
        log(f"  {task_key}: {len(samples)} samples")

    total = sum(len(s) for s in task_data.values())
    log(f"\n  {total} samples across {len(task_data)} tasks — {config.N_WORKERS} parallel workers\n")

    if not task_data:
        log("Nothing to run.")
    else:
        # ── Per-task state ─────────────────────────────────────────────────
        task_state = {
            k: {
                "y_true":     [],
                "y_pred":     [],
                "per_sample": [],
                "done":       [0],
                "lock":       threading.Lock(),
            }
            for k in task_data
        }
        global_done = [0]
        global_lock = threading.Lock()
        save_lock   = threading.Lock()

        # ── Submit round-robin across tasks so all run simultaneously ─────
        from itertools import zip_longest
        task_lists = [
            [(task_key, idx, text, label) for idx, (text, label) in enumerate(samples)]
            for task_key, samples in task_data.items()
        ]
        interleaved = [
            item
            for group in zip_longest(*task_lists)
            for item in group
            if item is not None
        ]

        with ThreadPoolExecutor(max_workers=config.N_WORKERS) as executor:
            futures = {
                executor.submit(process_one, task_key, TASKS[task_key], idx, text, label): (task_key, idx)
                for task_key, idx, text, label in interleaved
            }

            for future in as_completed(futures):
                task_key, _ = futures[future]
                r           = future.result()
                state       = task_state[task_key]
                n_total     = len(task_data[task_key])

                with state["lock"]:
                    state["done"][0] += 1
                    task_done = state["done"][0]

                    with global_lock:
                        global_done[0] += 1
                        g = global_done[0]

                    if r is None:
                        log(f"  [{g:>5}/{total}] {task_key} [{task_done}/{n_total}]  ERROR — skipped")
                    else:
                        state["y_true"].append(r["truth"])
                        state["y_pred"].append(r["pred"])
                        state["per_sample"].append(r["sample"])
                        log(
                            f"  [{g:>5}/{total}] {task_key} [{task_done}/{n_total}]"
                            f"  truth={r['truth']!r}  pred={r['pred']!r}  ({r['elapsed']:.1f}s)"
                        )

                    # Checkpoint
                    if task_done % SAVE_EVERY == 0:
                        with save_lock:
                            results[task_key] = {
                                "status":     "partial",
                                "completed":  task_done,
                                "per_sample": sorted(state["per_sample"], key=lambda x: x["id"]),
                            }
                            save_results(results)
                        log(f"  [CHECKPOINT] {task_key} — {task_done}/{n_total} saved.")

                    # Task complete
                    if task_done == n_total:
                        metrics = compute_metrics(
                            TASKS[task_key]["type"], state["y_true"], state["y_pred"]
                        )
                        with save_lock:
                            results[task_key] = {
                                "status":     "complete",
                                "metrics":    metrics,
                                "n_samples":  n_total,
                                "per_sample": sorted(state["per_sample"], key=lambda x: x["id"]),
                            }
                            save_results(results)
                        log(f"\n  *** {task_key} COMPLETE: {metrics} ***\n")

    # ── Final summary ──────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("  FINAL SUMMARY")
    log("="*60)
    for task_key, data in results.items():
        status  = data.get("status", "?")
        metrics = data.get("metrics", {})
        log(f"  {task_key:<14} {status:<10} {metrics}")

    log(f"\nResults written to: {RESULTS_FILE}")
    log(f"Thinking traces written to: {config.TRACES_LOG}")


if __name__ == "__main__":
    main()
