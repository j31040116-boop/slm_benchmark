"""
prompts.py — Task-specific prompt templates.

Each function receives a text string and returns the full prompt sent to the
model.  The model is instructed to:
  1. Reason inside <think>…</think> tags.
  2. Return a final answer inside [[result]]…[[/result]] tags.

Swap or extend these functions to experiment with different prompt styles.
"""


# ── Shared wrapper ────────────────────────────────────────────────────────────

def _wrap(instruction: str, text: str) -> str:
    return (
        f"{instruction}\n\n"
        f"Input:\n{text}\n\n"
        "Respond in this exact format:\n"
        "<think>step-by-step reasoning here</think>\n"
        "Answer: YOUR_ANSWER"
    )


# ── Per-task prompts ──────────────────────────────────────────────────────────

def fpb_prompt(text: str) -> str:
    return _wrap(
        "Classify the financial sentiment of the following sentence as exactly "
        "one of: negative, neutral, or positive.",
        text,
    )


def fiqasa_prompt(text: str) -> str:
    return _wrap(
        "Classify the financial sentiment of the following text as exactly "
        "one of: negative, neutral, or positive.",
        text,
    )


def finer_ord_prompt(text: str) -> str:
    return _wrap(
        "You are a Named Entity Recognition expert for financial documents. "
        "Does the following token belong to a named entity (such as an "
        "organisation, person, or location)? Reply with exactly one of: "
        "entity or O (if not an entity).",
        text,
    )


def convfinqa_prompt(text: str) -> str:
    return _wrap(
        "You are a financial analyst. Answer the following financial question "
        "with a single numeric value. Do not include units or extra words — "
        "just the number (e.g. 0.42 or -12.5).",
        text,
    )


def fomc_prompt(text: str) -> str:
    return _wrap(
        "Classify the monetary policy stance of the following FOMC text as "
        "exactly one of: dovish, neutral, or hawkish.",
        text,
    )


def ectsum_prompt(text: str) -> str:
    return _wrap(
        "Classify the overall sentiment of the following earnings call transcript "
        "summary as exactly one of: negative, neutral, or positive.",
        text,
    )


def acl18_prompt(text: str) -> str:
    return _wrap(
        "Classify the following financial tweet as exactly one of: bearish, neutral, or bullish.",
        text,
    )


# ── Dispatch table ────────────────────────────────────────────────────────────

PROMPT_FN = {
    "FPB":       fpb_prompt,
    "FiQA-SA":   fiqasa_prompt,
    "FiNER-ORD": finer_ord_prompt,
    "ConvFinQA": convfinqa_prompt,
    "FOMC":      fomc_prompt,
    "ECTSum":    ectsum_prompt,
    "ACL18":     acl18_prompt,
}


def get_prompt(task_key: str, text: str) -> str:
    """Return the formatted prompt for a given task."""
    fn = PROMPT_FN.get(task_key)
    if fn is None:
        raise ValueError(f"No prompt function registered for task '{task_key}'")
    return fn(text)
