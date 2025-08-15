# tests/test_openevals_automotive_dd.py
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
import requests

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

load_dotenv()

# --- Config helpers (robust to bad env values) ---
def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        print(f"[config] Warning: {name}='{raw}' is not a float; using default {default}")
        return float(default)

def coerce_float(value, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)

# --- Dataset & Judge config ---
DATASET_PATH = Path(os.getenv("DATASET_PATH", "datasets/sample_automotive_supply_chain.jsonl"))
USE_PRECOMPUTED_OUTPUTS = os.getenv("USE_PRECOMPUTED_OUTPUTS", "true").lower() == "true"
GLOBAL_MIN_SCORE = env_float("MIN_SCORE", 0.6)
JUDGE_MODEL = os.getenv("JUDGEMENT_MODEL", "openai:o3-mini")

# --- Ollama config (for dynamic outputs) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- JSONL logging config ---
LOG_PATH = Path(os.getenv("OPEN_EVAL_LOG_FILE", "logs/open_evals_runs.jsonl"))
SUITE_NAME = os.getenv("SUITE_NAME", "automotive_supply_chain")
LOG_TRUNCATE = os.getenv("LOG_TRUNCATE", "true").lower() == "true"
LOG_SCHEMA_VERSION = 2
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Judge (uses built-in prompt so OpenEvals can parse result) ---
judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,   # expects: inputs, outputs, reference_outputs
    model=JUDGE_MODEL,
    feedback_key="automotive_correctness",
)

# --- Helpers ---
def read_jsonl(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "input" not in row:
                raise ValueError(f"Line {i}: missing 'input'")
            row["_line"] = i
            yield row

def normalize_reference(row):
    """
    Accepts common keys: ideal | expected | reference_outputs | reference | answer
    Returns (ref_list_for_judge, expected_text_for_log)
    """
    candidate = (
        row.get("ideal")
        or row.get("expected")
        or row.get("reference_outputs")
        or row.get("reference")
        or row.get("answer")
    )
    if candidate is None:
        return [], ""
    if isinstance(candidate, list):
        ref_list = [str(x) for x in candidate]
    else:
        ref_list = [str(candidate)]
    # Pretty join for HTML modal readability
    expected_text = "\n---\n".join(ref_list)
    return ref_list, expected_text

def case_id(row):
    meta = row.get("metadata") or {}
    if meta.get("topic"):
        return str(meta["topic"])
    return f"case-{hashlib.sha1(row['input'].encode()).hexdigest()[:8]}"

def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@pytest.fixture(scope="session", autouse=True)
def _init_log():
    # Truncate once per test session (unless disabled)
    if LOG_TRUNCATE and LOG_PATH.exists():
        LOG_PATH.unlink()
    LOG_PATH.touch(exist_ok=True)
    yield

# --- System-under-test output via Ollama (when not using precomputed outputs) ---
def generate_output(task_input: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": task_input,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

# --- Load data & parametrize ---
try:
    DATA = list(read_jsonl(DATASET_PATH))
except FileNotFoundError as e:
    pytest.skip(str(e) + " (set DATASET_PATH to a valid dataset)", allow_module_level=True)

@pytest.mark.parametrize("row", DATA, ids=[case_id(r) for r in DATA])
def test_case(row):
    # OpenEvals judge needs an OpenAI key for the judge model
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required"

    inputs = row["input"]
    ref_list, expected_text = normalize_reference(row)

    if not ref_list:
        raise AssertionError("No reference/expected answer found in row. Add 'ideal' or 'expected'.")

    # Provide outputs (precomputed vs generated)
    used_precomputed = bool(USE_PRECOMPUTED_OUTPUTS and "output" in row)
    outputs = row.get("output") if used_precomputed else generate_output(inputs)

    # Evaluate with OpenEvals judge
    result = judge(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=ref_list,
    )

    # Score & feedback
    score = coerce_float(result.get("score"), 0.0)
    # OpenEvals may return either 'comment' or a structured 'feedback' dict
    feedback = result.get("comment", "")
    if not feedback:
        fb = result.get("feedback") or {}
        # try the custom key first, then any string fallback
        feedback = fb.get("automotive_correctness") or (fb if isinstance(fb, str) else "")

    threshold = coerce_float(row.get("min_score"), GLOBAL_MIN_SCORE)
    meta = row.get("metadata") or {}
    topic = meta.get("topic")
    difficulty = meta.get("difficulty")
    test_name = case_id(row)

    passed = bool(score >= threshold)

    # Console output (helpful in CI)
    print(f"\n== {test_name} ==")
    print(f"score={score:.3f} threshold={threshold:.3f} passed={passed}")
    if topic or difficulty:
        print(f"topic={topic} difficulty={difficulty}")
    if feedback:
        print(f"judge_feedback={feedback}")

    # --- JSONL log line (HTML-friendly fields) ---
# --- JSONL log line (HTML-friendly fields) ---
    log_record = {
        "schema_version": LOG_SCHEMA_VERSION,
        "framework": "OpenEvals",
        "suite": SUITE_NAME,
        "case_id": test_name,

        # Table columns
        "difficulty": difficulty,
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "pass": passed,  # some viewers expect 'pass' instead of 'passed'

        # Modal/details
        "input": inputs,
        "expected": expected_text,   # <-- HTML looks for this
        "ideal": ref_list,           # keep the raw list too
        "output": outputs,

        # Extras for filtering/debugging
        "judge_feedback": feedback,
        "model": JUDGE_MODEL,
        "judgement_model": JUDGE_MODEL,
        "judgement_reason": feedback,
        "generator_model": (None if used_precomputed else OLLAMA_MODEL),
        "sut_model": (None if used_precomputed else OLLAMA_MODEL),
        "used_precomputed": used_precomputed,
        "topic": topic,
        "metadata": meta,
        "dataset_line": row.get("_line"),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_jsonl(LOG_PATH, log_record)

    # Assert last so failures are still logged
    assert passed, f"Score {score:.3f} < {threshold:.3f} — {feedback}"
