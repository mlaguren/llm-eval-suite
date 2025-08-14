import json
import os
import re
import string
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

# --- Config ---
DATA_FILE = os.getenv("DATASET_PATH", "samples_automotive_supply_chain.jsonl")
MODEL_RUNNER = os.getenv("MODEL_RUNNER", "ollama")  # ollama | http | noop
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
APP_ENDPOINT = os.getenv("APP_ENDPOINT", "http://localhost:8000/generate")
CORRECTNESS_THRESHOLD = float(os.getenv("CORRECTNESS_THRESHOLD", "0.5"))
LOG_FILE = os.getenv("EVAL_LOG_FILE", "deepeval_runs.jsonl")
TRIM_PREVIEW = int(os.getenv("EVAL_PREVIEW_CHARS", "600"))
# --- Judge config ---
JUDGEMENT_MODEL = os.getenv("DEEPEVAL_JUDGEMENT_MODEL", "gpt-4o-mini")


# --- Helpers ---
def load_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_model_output(prompt: str) -> str:
    if MODEL_RUNNER == "ollama":
        try:
            res = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return res.stdout.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama call failed: {e.stderr.decode('utf-8')}")
    elif MODEL_RUNNER == "http":
        import requests
        r = requests.post(APP_ENDPOINT, json={"input": prompt}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("output") or data.get("text") or "").strip()
    elif MODEL_RUNNER == "noop":
        return ""
    else:
        raise ValueError(f"Unknown MODEL_RUNNER: {MODEL_RUNNER}")

# basic english stopwords for diff readability
_STOP = {
    "a","an","the","and","or","if","then","else","for","of","on","in","to","from","by",
    "with","without","is","are","was","were","be","been","being","that","this","these",
    "those","it","its","as","at","into","about","than","so","such","but","not","no"
}
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

def _tokens(s: str):
    s = _PUNCT_RE.sub(" ", s.lower())
    toks = [t for t in s.split() if t and t not in _STOP and not t.isdigit()]
    return set(toks)

def token_diff(actual: str, ideal: str):
    a = _tokens(actual)
    b = _tokens(ideal)
    unexpected = sorted(a - b)   # present in actual, not in ideal
    missing = sorted(b - a)      # present in ideal, not in actual
    return unexpected[:20], missing[:20]

def write_log(rec: dict):
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _case_id(row: dict, idx: int) -> str:
    topic = (row.get("metadata") or {}).get("topic", "")
    head = (row.get("input",""))[:40].replace("\n", " ")
    parts = [f"{idx:03d}"]
    if topic:
        parts.append(topic)
    parts.append(head + ("…" if len(row.get("input","")) > 40 else ""))
    return " | ".join(parts)

def _extract_score_and_reason(measure_result, metric_obj):
    """
    Be defensive across DeepEval versions:
    - score: prefer result.score, fall back to metric.score
    - reason/explanation: check many common fields (result then metric), and nested dicts
    """
    candidates_text_fields = [
        "reason", "reasoning", "explanation", "feedback", "justification",
        "grade_comment", "notes", "message"
    ]

    # score
    score = None
    for source in (measure_result, metric_obj):
        if source is None:
            continue
        s = getattr(source, "score", None)
        if isinstance(s, (int, float)):
            score = float(s)
            break

    # reason (string)
    reason = None
    for source in (measure_result, metric_obj):
        if source is None:
            continue
        # direct attributes
        for name in candidates_text_fields:
            val = getattr(source, name, None)
            if isinstance(val, str) and val.strip():
                reason = val.strip()
                break
        if reason:
            break
        # common containers
        for container_name in ("evaluation_output", "raw_output", "raw"):
            container = getattr(source, container_name, None)
            if isinstance(container, dict):
                for name in candidates_text_fields + ["reason"]:
                    val = container.get(name)
                    if isinstance(val, str) and val.strip():
                        reason = val.strip()
                        break
            if reason:
                break

    # raw (best-effort) for debugging
    raw = None
    for source in (measure_result, metric_obj):
        if source is None:
            continue
        raw = getattr(source, "raw_output", None) or getattr(source, "evaluation_output", None)
        if raw:
            break

    return score, reason, raw


# Build judge metric (LLM-as-judge)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=CORRECTNESS_THRESHOLD,
    model=JUDGEMENT_MODEL,
)

# Load samples
SAMPLES = list(load_jsonl(DATA_FILE))

@pytest.mark.parametrize("row", SAMPLES, ids=[_case_id(r, i) for i, r in enumerate(SAMPLES, start=1)])
def test_automotive_supply_chain_case(row):
    prompt = row["input"]
    expected = row["ideal"]
    meta = row.get("metadata") or {}
    retrieval_context = row.get("retrieval_context", [])

    # 1) get model output
    actual = get_model_output(prompt)

    # 2) evaluate with DeepEval's judge
    tc = LLMTestCase(
        input=prompt,
        actual_output=actual,
        expected_output=expected,
        retrieval_context=retrieval_context,
    )
    # capture the measure result (if available) to extract fields robustly
    measure_result = correctness_metric.measure(tc)
    score, reason, raw = _extract_score_and_reason(measure_result, correctness_metric)

    # 3) token-level diff (for logs & debugging)
    unexpected, missing = token_diff(actual, expected)

    # 4) write a structured log row (always)
    write_log({
        "ts": datetime.utcnow().isoformat() + "Z",
        "runner": MODEL_RUNNER,
        "generator_model": OLLAMA_MODEL if MODEL_RUNNER == "ollama" else None,
        "judgement_model": JUDGEMENT_MODEL,
        "topic": meta.get("topic"),
        "difficulty": meta.get("difficulty"),
        "input": (prompt[:TRIM_PREVIEW] + ("…" if len(prompt) > TRIM_PREVIEW else "")),
        "expected": (expected[:TRIM_PREVIEW] + ("…" if len(expected) > TRIM_PREVIEW else "")),
        "actual": (actual[:TRIM_PREVIEW] + ("…" if len(actual) > TRIM_PREVIEW else "")),
        "score": score,
        "threshold": CORRECTNESS_THRESHOLD,
        "judge_reason": reason,
        "judge_raw": raw,  # optional, can be large; comment out if you prefer slim logs
        "retrieval_context": retrieval_context,
        "diff": {
            "unexpected_tokens": unexpected,
            "missing_tokens": missing,
        },
        "meta": meta,
    })

    # 5) On failure, print helpful diffs so they appear under -q
    passed = (score is not None) and (score >= CORRECTNESS_THRESHOLD)
    if not passed:
        print("\n--- Failure details ---")
        print(f"Score: {score} (threshold {CORRECTNESS_THRESHOLD})")
        if meta:
            print(f"Meta: topic={meta.get('topic')}, difficulty={meta.get('difficulty')}")
        if reason:
            print(f"Judge note: {reason}")
        a_preview = (actual[:TRIM_PREVIEW] + ("…" if len(actual) > TRIM_PREVIEW else ""))
        e_preview = (expected[:TRIM_PREVIEW] + ("…" if len(expected) > TRIM_PREVIEW else ""))
        print(f"\nExpected:\n{e_preview}")
        print(f"\nActual:\n{a_preview}")
        if unexpected:
            print(f"\nUnexpected tokens (in actual, not in expected): {unexpected}")
        if missing:
            print(f"Missing tokens (in expected, not in actual): {missing}")
        print("\n-----------------------")

    # 6) final assertion (raises on fail)
    assert passed, f"Correctness score {score} below threshold {CORRECTNESS_THRESHOLD}"
