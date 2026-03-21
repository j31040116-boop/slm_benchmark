# SLM Financial Benchmark

Evaluation harness for small language models (SLMs) on financial NLP tasks, using [Ollama](https://ollama.com) for local inference.

**Current model under evaluation:** `LFM2.5-Thinking 1.2B` (`lfm2.5-thinking:latest`)

---

## Tasks

| Task | Dataset | Type | Metric | Samples |
|------|---------|------|--------|---------|
| FPB | Financial PhraseBank | Sentiment (3-class) | Accuracy, Macro-F1 | 1,000 |
| FiQA-SA | FiQA Sentiment Analysis | Sentiment (3-class) | Accuracy, Macro-F1 | 235 |
| FiNER-ORD | Financial NER (token-level) | Binary classification | Accuracy, Macro-F1 | 1,000 |
| ConvFinQA | Conversational Financial QA | Numeric exact match | Exact Match % | 1,490 |
| FOMC | FOMC Communication Stance | Stance (3-class) | Accuracy, Macro-F1 | 496 |
| ECTSum | Earnings Call Transcript Sentiment | Sentiment (3-class) | Accuracy, Macro-F1 | 506 |
| ACL18 | Twitter Financial News Sentiment | Sentiment (3-class) | Accuracy, Macro-F1 | 2,388 |

---

## Setup

### Prerequisites

- Python 3.13+ (stable release — **do not use alpha/pre-release Python**)
- [Ollama](https://ollama.com) installed and running
- LFM2.5-Thinking model pulled: `ollama pull lfm2.5-thinking:latest`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama (separate terminal)

```bash
ollama serve
```

---

## Running

```bash
# Run all tasks (full dataset)
python slm_eval.py

# Run specific tasks only
python slm_eval.py --tasks FPB ACL18

# Debug mode — first 3 samples per task (set DEBUG_MODE=True in config.py or:)
python slm_eval.py --tasks FPB --no-resume

# Ignore existing results and rerun from scratch
python slm_eval.py --no-resume
```

Results are saved incrementally to `benchmark_results.json` (checkpointed every 10 samples). If the run is interrupted, re-running will resume from where it left off.

---

## Visualizations

After a run completes, generate charts:

```bash
python visualize.py
```

Output charts saved to `charts/`:
- `line_accuracy_vs_f1.png` — Accuracy vs macro-F1 per task (line graph with gap shading)
- `heatmap_metrics.png` — Metric heatmap across all tasks
- `pred_distribution.png` — Prediction distribution per task
- `sample_timing.png` — Per-sample inference time

---

## Configuration

Edit [config.py](config.py) to change model, workers, or token limits:

```python
MODEL_NAME = "lfm2.5-thinking:latest"   # Ollama model tag
N_WORKERS  = 6       # parallel inference threads
MAX_TOKENS = 1536    # tokens for model output (includes thinking trace)
NUM_CTX    = 2048    # context window size
DEBUG_MODE = False   # True = only 3 samples per task
```

**Important:** `MAX_TOKENS` must be at least `1536` for thinking models. Lower values cause the thinking trace to consume all available tokens, leaving the model answer empty.

---

## File Structure

```
slm_benchmark/
├── slm_eval.py          # Main evaluation harness
├── config.py            # Model settings, task registry
├── prompts.py           # Per-task prompt templates
├── parsers.py           # Answer extraction from model output
├── metrics.py           # Accuracy, macro-F1, exact match
├── visualize.py         # Chart generation
├── organize_results.py  # Restructures results JSON
├── requirements.txt     # Python dependencies
├── benchmark_results.json  # Results (auto-saved, resume-safe)
└── charts/              # Generated visualizations
```

---

## SSH / Remote Execution

To run on another machine via SSH:

1. Push this repo and pull it on the remote machine
2. Install Ollama and pull the model: `ollama pull lfm2.5-thinking:latest`
3. Create a virtual environment with Python 3.13+:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   pip install -r requirements.txt
   ```
4. Run: `ollama serve &` then `python slm_eval.py`

The `benchmark_results.json` will be populated as results come in. Results are checkpointed every 10 samples so it's safe to interrupt and resume.

---

## Known Caveats

- **Accuracy vs macro-F1:** All classification tasks have class imbalance (e.g., neutral-heavy datasets). Accuracy appears high but macro-F1 is the reliable metric. A model predicting "neutral" for everything can achieve 60%+ accuracy with ~0.15 macro-F1.
- **Thinking trace overhead:** LFM2.5-Thinking uses chain-of-thought before answering. Each sample takes ~30–40 seconds. Full benchmark (~7,100 samples) takes ~25 hours wall time on a single GPU at 6 workers.
- **ConvFinQA difficulty:** Long financial table inputs combined with exact-match numeric grading make this task hard for 1.2B models.

---

## Results

Results are stored in `benchmark_results.json` in organized format with `run_info`, `summary`, and `per_task_detail` sections.

> **Note:** Results below require a full rerun with the current fixed codebase. Several bugs were discovered and fixed after initial runs (token truncation causing empty predictions, FiNER-ORD prompt/label mismatch, normalise_label regex bug). ACL18 has been rerun with fixes applied; other tasks are pending rerun.

| Task | n | Accuracy | Macro-F1 | Status |
|------|---|----------|----------|--------|
| FPB | 1,000 | 0.616 | 0.105 | needs rerun |
| FiQA-SA | 235 | 0.583 | 0.264 | needs rerun |
| FiNER-ORD | 1,000 | 0.768 | 0.067 | needs rerun |
| ConvFinQA | 1,490 | — | — | 8.9% EM, needs rerun |
| FOMC | 496 | 0.389 | 0.102 | needs rerun |
| ECTSum | 506 | 0.480 | 0.120 | needs rerun |
| ACL18 | 2,388 | 0.484 | 0.045 | complete (fixed run) |
