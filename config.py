"""
config.py — Central configuration for the SLM evaluation harness.
Edit this file to swap models, adjust limits, or toggle debug mode.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "lfm2.5-thinking:latest"   # Ollama model tag  ← edit to your tag
OLLAMA_HOST = "http://localhost:11434"  # Ollama server URL
THINK_MODE = True                       # thinking model — must stay True

# ── Run control ──────────────────────────────────────────────────────────────
DEBUG_MODE = False       # True → only first 3 rows per task
SAVE_EVERY = 10          # auto-save checkpoint every N samples
MAX_SAMPLES = None       # None = full dataset; int = hard cap per task
N_WORKERS  = 6           # increased from 4 — Windows/GPU handles more concurrency
MAX_TOKENS = 1536        # 1024 caused truncation on ACL18; 1536 gives thinking trace room
NUM_CTX    = 2048        # context window — default Ollama is 4096; 2048 is enough and faster

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_FILE = "benchmark_results.json"
TRACES_LOG   = "traces.log"

# ── Task registry ─────────────────────────────────────────────────────────────
# Each entry: (task_key, hf_path, hf_name_or_None, split, label_col, text_col)
# label_col / text_col may be overridden per-task in the loader when the
# schema is non-standard (see slm_eval.py).
TASKS = {
    "FPB": {
        "hf_path":  "warwickai/financial_phrasebank_mirror",
        "hf_name":  None,
        "split":    "train",
        "type":     "classification",   # accuracy + macro-F1
        "text_col": "sentence",
        "label_col":"label",
        # FPB labels: 0=negative, 1=neutral, 2=positive
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "max_samples": 1000,
    },
    "FiQA-SA": {
        "hf_path":  "ChanceFocus/flare-fiqasa",
        "hf_name":  None,
        "split":    "test",
        "type":     "classification",
        "text_col": "text",
        "label_col":"answer",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
    },
    "FiNER-ORD": {
        "hf_path":    "gtfintechlab/finer-ord",
        "hf_name":    None,
        "split":      "test",
        "type":       "classification",
        "text_col":   "gold_token",
        "label_col":  "gold_label",
        "max_samples": 1000,   # dataset has 30k+ token-level rows — cap it
    },
    "ConvFinQA": {
        "hf_path":  "AdaptLLM/finance-tasks",
        "hf_name":  "ConvFinQA",
        "split":    "test",
        "type":     "exact_match",      # numeric EM
        "text_col": "input",
        "label_col":"label",
    },
    "FOMC": {
        "hf_path":  "gtfintechlab/fomc_communication",
        "hf_name":  None,
        "split":    "test",
        "type":     "classification",
        "text_col": "sentence",
        "label_col":"label",
        "label_map": {0: "dovish", 1: "hawkish", 2: "neutral"},
    },
    "ECTSum": {
        "hf_path":  "nickmuchi/financial-classification",
        "hf_name":  None,
        "split":    "test",
        "type":     "classification",
        "text_col": "text",
        "label_col":"labels",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
    },
    "ACL18": {
        "hf_path":  "zeroshot/twitter-financial-news-sentiment",
        "hf_name":  None,
        "split":    "validation",
        "type":     "classification",
        "text_col": "text",
        "label_col":"label",
        "label_map": {0: "bearish", 1: "bullish", 2: "neutral"},
    },
}
