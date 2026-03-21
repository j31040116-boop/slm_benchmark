"""
parsers.py — Extraction of think-traces and final answers from model output.
"""

import re
import logging
from pathlib import Path
from config import TRACES_LOG

# One file-appending logger for all traces
_trace_logger = logging.getLogger("traces")
_trace_logger.setLevel(logging.DEBUG)
if not _trace_logger.handlers:
    _fh = logging.FileHandler(TRACES_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _trace_logger.addHandler(_fh)

# ── Regex patterns ────────────────────────────────────────────────────────────

_THINK_RE   = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE  = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def extract_think(raw: str, task: str = "", sample_id: int = 0) -> str:
    """Pull <think>…</think> content and append to traces.log."""
    match = _THINK_RE.search(raw)
    trace = match.group(1).strip() if match else ""
    _trace_logger.debug(
        "=== [%s | sample %d] ===\n%s\n", task, sample_id, trace or "(no trace)"
    )
    return trace


def extract_result(raw: str) -> str:
    """
    Extract the answer from the model response.
    Primary: looks for 'Answer: X' after the think block.
    Fallback: last non-empty line, then last word.
    """
    # Strip think block first
    cleaned = _THINK_RE.sub("", raw).strip()

    m = _ANSWER_RE.search(cleaned)
    if m:
        # Take only the first line; strip placeholder if model echoed it
        answer = m.group(1).split("\n")[0].strip()
        answer = re.sub(r"(?i)your[_\s]?answer\s*", "", answer).strip()
        if answer:
            return answer

    # Fallback: last non-empty line
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) <= 40:
            return last_line
        words = last_line.rstrip(".,;:").split()
        if words:
            return words[-1]
    return cleaned


def parse_float(text: str) -> float | None:
    """Extract the first float-like token from text."""
    m = re.search(r"-?\d+\.?\d*", text)
    return float(m.group()) if m else None


def normalise_label(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for label comparison."""
    return re.sub(r"[^a-z0-9\-]", "", text.lower()).strip()
