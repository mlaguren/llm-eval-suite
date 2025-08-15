import json
import os
import re
import string
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Config ---
# JSONL dataset shipped in this repo
DATA_FILE = os.getenv("DATASET_PATH", "data_sets/samples_automotive_supply_chain.jsonl")
MODEL_RUNNER = os.getenv("MODEL_RUNNER", "ollama")  # ollama | http | noop | openai
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_ENDPOINT = os.getenv("APP_ENDPOINT", "http://localhost:8000/generate")
CORRECTNESS_THRESHOLD = float(os.getenv("CORRECTNESS_THRESHOLD", "0.5"))
LOG_FILE = os.getenv("OPENAI_EVAL_LOG_FILE", "logs/openai_evals_runs.jsonl")
TRIM_PREVIEW = int(os.getenv("EVAL_PREVIEW_CHARS", "600"))

# --- Judge config ---
JUDGEMENT_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


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
        except FileNotFoundError as e:
            raise RuntimeError("Ollama binary not found. Install ollama or set MODEL_RUNNER to 'http'/'openai'/'noop'.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama call failed: {e.stderr.decode('utf-8')}") from e
    elif MODEL_RUNNER == "openai":
        if not client:
            raise RuntimeError("OpenAI client not initialized - check OPENAI_API_KEY")
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
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


def openai_judge_correctness(
    prompt: str, 
    expected: str, 
    actual: str, 
    retrieval_context: List[str] = None
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Use OpenAI API to judge correctness of the actual output against expected output.
    Returns (score, reason, raw_response)
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized - check OPENAI_API_KEY")
        
    context_str = ""
    if retrieval_context:
        context_str = f"\n\nRetrieved Context:\n" + "\n".join(f"- {ctx}" for ctx in retrieval_context)

    judge_prompt = f"""You are an expert evaluator. Please evaluate the correctness of the actual output compared to the expected output for the given input.

Input Query:
{prompt}
{context_str}

Expected Output:
{expected}

Actual Output:
{actual}

Evaluation Criteria:
- Rate correctness on a scale from 0.0 to 1.0
- 1.0 = Perfect match or semantically equivalent
- 0.8-0.9 = Mostly correct with minor issues
- 0.5-0.7 = Partially correct but missing key information
- 0.2-0.4 = Major inaccuracies or missing critical content
- 0.0-0.1 = Completely incorrect or irrelevant

Provide your response in this exact JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your evaluation>",
    "key_issues": ["<list of specific issues found>"],
    "strengths": ["<list of what the actual output did well>"]
}}"""

    try:
        response = client.chat.completions.create(
            model=JUDGEMENT_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=1000,
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            result = json.loads(content)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            return score, reasoning, result
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract score with regex if JSON parsing fails
            score_match = re.search(r'"score":\s*([0-9]*\.?[0-9]+)', content)
            score = float(score_match.group(1)) if score_match else 0.0
            return score, content, {"raw_content": content, "parse_error": str(e)}

    except Exception as e:
        print(f"Warning: OpenAI judge evaluation failed: {str(e)}")
        return 0.0, f"Evaluation failed: {str(e)}", {"error": str(e)}


# Load samples
SAMPLES = list(load_jsonl(DATA_FILE))


@pytest.mark.parametrize("row", SAMPLES, ids=[_case_id(r, i) for i, r in enumerate(SAMPLES, start=1)])
def test_automotive_supply_chain_case(row):
    prompt = row["input"]
    expected = row["ideal"]
    meta = row.get("metadata") or {}
    retrieval_context = row.get("retrieval_context", [])

    # Skip when judge cannot run or generator is unavailable
    if client is None:
        pytest.skip("Skipping OpenAI eval: OPENAI_API_KEY not set")
    if MODEL_RUNNER == "ollama" and shutil.which("ollama") is None:
        pytest.skip("Skipping: ollama binary not found on PATH")

    # 1) get model output
    actual = get_model_output(prompt)

    # 2) evaluate with OpenAI judge
    score, reason, raw = openai_judge_correctness(prompt, expected, actual, retrieval_context)

    # 3) token-level diff (for logs & debugging)
    unexpected, missing = token_diff(actual, expected)

    # 4) write a structured log row (always)
    write_log({
        "ts": datetime.utcnow().isoformat() + "Z",
        "runner": MODEL_RUNNER,
        "generator_model": OLLAMA_MODEL if MODEL_RUNNER == "ollama" else OPENAI_MODEL if MODEL_RUNNER == "openai" else None,
        "judgement_model": JUDGEMENT_MODEL,
        "topic": meta.get("topic"),
        "difficulty": meta.get("difficulty"),
        "input": (prompt[:TRIM_PREVIEW] + ("…" if len(prompt) > TRIM_PREVIEW else "")),
        "expected": (expected[:TRIM_PREVIEW] + ("…" if len(expected) > TRIM_PREVIEW else "")),
        "actual": (actual[:TRIM_PREVIEW] + ("…" if len(actual) > TRIM_PREVIEW else "")),
        "score": score,
        "threshold": CORRECTNESS_THRESHOLD,
        "judge_reason": reason,
        "judge_raw": raw,
        "retrieval_context": retrieval_context,
        "diff": {
            "unexpected_tokens": unexpected,
            "missing_tokens": missing,
        },
        "meta": meta,
    })

    # 5) On failure, print helpful diffs so they appear under -q
    passed = score >= CORRECTNESS_THRESHOLD
    if not passed:
        print("\n--- Failure details ---")
        print(f"Score: {score} (threshold {CORRECTNESS_THRESHOLD})")
        if meta:
            print(f"Meta: topic={meta.get('topic')}, difficulty={meta.get('difficulty')}")
        if reason:
            print(f"Judge reasoning: {reason}")
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


if __name__ == "__main__":
    # Optional: Run a quick test of the judge function
    print("Testing OpenAI judge function...")
    test_score, test_reason, test_raw = openai_judge_correctness(
        "What is 2+2?",
        "4",
        "The answer is 4"
    )
    print(f"Test score: {test_score}")
    print(f"Test reason: {test_reason}")