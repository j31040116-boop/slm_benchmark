"""
audit.py — Full pre-run audit of dataset labels, prompt alignment, and parser bugs.
"""
from datasets import load_dataset
from collections import Counter
import re

SEP = "=" * 60

# ── 1. FiQA-SA — check answer column ─────────────────────────────────────────
print(SEP)
print("1. FiQA-SA — answer column format")
ds = load_dataset("ChanceFocus/flare-fiqasa", split="test")
answers = ds["answer"]
print(f"   Type of answer values: {type(answers[0])}")
print(f"   Unique values: {Counter(answers).most_common()}")
print(f"   Sample[0]: answer={answers[0]}  text={ds[0]['text'][:60]}")

# ── 2. FOMC — verify label mapping ───────────────────────────────────────────
print(SEP)
print("2. FOMC — label mapping verification")
ds = load_dataset("gtfintechlab/fomc_communication", split="test")
label_counts = Counter(ds["label"])
print(f"   Label distribution: {dict(label_counts)}")
mapping = {0: "dovish", 1: "hawkish", 2: "neutral"}
for lbl, name in mapping.items():
    rows = [r for r in ds if r["label"] == lbl]
    if rows:
        print(f"   label={lbl} ({name}): '{rows[0]['sentence'][:70]}'")

# ── 3. FiNER-ORD — check gold_label values ───────────────────────────────────
print(SEP)
print("3. FiNER-ORD — gold_label distribution")
ds = load_dataset("gtfintechlab/finer-ord", split="test")
label_counts = Counter(ds["gold_label"])
print(f"   Label distribution: {dict(label_counts)}")
print("   Sample rows:")
for i in range(6):
    row = ds[i]
    print(f"   token={repr(row['gold_token'])}  gold_label={row['gold_label']}")

# ── 4. ECTSum — check 'labels' column ────────────────────────────────────────
print(SEP)
print("4. ECTSum — labels column format")
ds = load_dataset("nickmuchi/financial-classification", split="test")
lvals = [ds[i]["labels"] for i in range(5)]
print(f"   Type: {type(lvals[0])}")
print(f"   Sample values: {lvals}")
print(f"   Distribution: {Counter(ds['labels']).most_common()}")

# ── 5. ConvFinQA — check label format ────────────────────────────────────────
print(SEP)
print("5. ConvFinQA — label column format")
ds = load_dataset("AdaptLLM/finance-tasks", name="ConvFinQA", split="test")
labels = [ds[i]["label"] for i in range(8)]
print(f"   Type: {type(labels[0])}")
print(f"   Sample labels: {labels}")

# ── 6. Parser bug — normalise_label keeps periods ────────────────────────────
print(SEP)
print("6. Parser — normalise_label period bug")
def normalise_label(text):
    return re.sub(r"[^a-z0-9\-\.]", "", text.lower()).strip()

tests = ["positive.", "negative.", "neutral,", "bullish!", "Answer: bearish", "positive"]
for t in tests:
    print(f"   normalise_label({repr(t)}) = {repr(normalise_label(t))}")

# ── 7. FiNER-ORD prompt vs label mismatch ────────────────────────────────────
print(SEP)
print("7. FiNER-ORD — prompt asks ORG/PER/LOC/O but harness maps to entity/O")
print("   Prompt says: classify as one of: ORG, PER, LOC, or O")
print("   Harness expects pred to match: 'entity' or 'o'")
print("   If model says 'ORG' -> normalise='org' != 'entity' -> ALWAYS WRONG")

# ── 8. FPB label map sanity check ────────────────────────────────────────────
print(SEP)
print("8. FPB — warwickai mirror label distribution")
ds = load_dataset("warwickai/financial_phrasebank_mirror", split="train")
print(f"   Distribution: {Counter(ds['label']).most_common()}")
print(f"   Config map: 0=negative, 1=neutral, 2=positive")
print(f"   Sample label=0: {next(r for r in ds if r['label']==0)['sentence'][:60]}")
print(f"   Sample label=2: {next(r for r in ds if r['label']==2)['sentence'][:60]}")

print(SEP)
print("AUDIT COMPLETE")
