import os
from pathlib import Path
import braintrust as bt
from openai import OpenAI
import json
import pytest

# Always load .env if present (even in CI) so DATASET_PATH is picked up locally
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# --- Config ---
DATA_FILE = os.getenv("DATASET_PATH", "data_sets/sample_automotive_supply_chain.jsonl")
MODEL_RUNNER = os.getenv("MODEL_RUNNER", "openai")  # ollama | http | noop | openai
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_ENDPOINT = os.getenv("APP_ENDPOINT", "http://localhost:8000/generate")
CORRECTNESS_THRESHOLD = float(os.getenv("CORRECTNESS_THRESHOLD", "0.5"))

# Use JUDGEMENT_MODEL from env (for consistency with other evals) or fallback to OPENAI_JUDGE_MODEL
JUDGEMENT_MODEL = os.getenv("JUDGEMENT_MODEL") or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CI/automation: configure project + API key without braintrust init
BT_API_KEY = os.getenv("BRAINTRUST_API_KEY")
BT_PROJECT = os.getenv("BRAINTRUST_PROJECT", "automotive-supply-chain")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# --- Model runner ---
def get_model_output(prompt: str) -> str:
    if MODEL_RUNNER == "ollama":
        import subprocess
        res = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return res.stdout.decode("utf-8").strip()
    elif MODEL_RUNNER == "openai":
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
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


# --- Evaluator ---
def correctness_eval(example, output, expected=None):
    """
    Braintrust scorer function - returns a dictionary with score and metadata
    In Braintrust 0.2.3, parameters may be passed differently
    """
    # Handle different parameter formats from Braintrust
    if isinstance(example, dict):
        # example is a dict with input/expected
        input_text = example.get("input", "")
        expected_text = example.get("expected", expected or "")
    else:
        # example is just the input string
        input_text = str(example)
        expected_text = expected or ""
    
    if not client:
        return {"score": 0.0, "metadata": {"error": "OpenAI API not configured"}}

    if not expected_text:  # Handle missing expected values
        return {"score": 0.0, "metadata": {"error": "No expected value provided"}}

    prompt = f"""You are an expert evaluator. Evaluate how well the actual output matches the expected output for the given input.

Input: {input_text}
Expected: {expected_text}
Actual: {output}

Rate the correctness on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect match, semantically equivalent, or equally valid alternative answers
- 0.8 = Very good, minor differences or valid alternative that addresses the question fully
- 0.6 = Good, some important information missing but generally correct
- 0.4 = Partially correct
- 0.2 = Mostly incorrect but some relevant content
- 0.0 = Completely wrong or irrelevant

For factual questions with multiple valid answers (like naming companies in an industry), consider whether the actual answer provides valid examples that fulfill the request, even if they differ from the expected answer.

Respond with only a JSON object with fields "score" (number between 0.0 and 1.0) and "reasoning" (brief explanation)."""

    try:
        resp = client.chat.completions.create(
            model=JUDGEMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )

        content = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            reasoning = parsed.get("reasoning", "")
            return {"score": score, "metadata": {"reasoning": reasoning}}
        except Exception as parse_error:
            import re
            m = re.search(r"([0-9]*\.?[0-9]+)", content)
            score = float(m.group(1)) if m else 0.0
            score = max(0.0, min(1.0, score))
            return {
                "score": score,
                "metadata": {"reasoning": content, "parse_error": str(parse_error)},
            }
    except Exception as e:
        return {"score": 0.0, "metadata": {"error": str(e)}}


# --- Dataset ---
data_path = Path(DATA_FILE)

if not data_path.exists():
    pytest.skip(f"Dataset file not found: {data_path.resolve()} (set DATASET_PATH)", allow_module_level=True)

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON on line {idx}, skipping: {e}")
            continue

        # Validate required fields
        if "input" not in obj:
            print(f"Warning: Missing 'input' key in dataset record line {idx}, skipping")
            continue
        if "ideal" not in obj:
            print(f"Warning: Missing 'ideal' key in dataset record line {idx}, skipping")
            continue

        # Ensure both input and ideal are non-empty strings
        if not obj["input"] or not obj["ideal"]:
            print(f"Warning: Empty input or ideal on line {idx}, skipping")
            continue

        # Convert to Braintrust format
        braintrust_record = {
            "input": str(obj["input"]).strip(),
            "expected": str(obj["ideal"]).strip(),
        }

        # Add any metadata fields
        for key, value in obj.items():
            if key not in ["input", "ideal"]:
                braintrust_record[key] = value

        records.append(braintrust_record)

print(f"Loaded {len(records)} valid records from dataset")

if not records:
    pytest.skip(f"No valid records loaded from dataset: {data_path.resolve()}", allow_module_level=True)


# Alternative: Individual test per record (unchanged)
@pytest.mark.parametrize("record", records)
def test_individual_record(record):
    """Test each record individually (creates separate tests for each record)"""
    print(f"\n=== INDIVIDUAL RECORD TEST ===")
    print(f"Testing record: {record.get('input', '')[:100]}...")
    print(f"Expected: {record.get('expected', '')[:50]}...")
    
    # Get model output for this specific record
    output = get_model_output(record["input"])
    print(f"Model output: {output[:50]}...")

    # Evaluate correctness using the scorer
    result = correctness_eval(record, output)
    score = result["score"]
    reasoning = result.get("metadata", {}).get("reasoning", "No reasoning provided")

    print(f"Score: {score:.3f}, Threshold: {CORRECTNESS_THRESHOLD}")
    
    # Assert the score meets the threshold
    assert score >= CORRECTNESS_THRESHOLD, (
        f"Score {score:.3f} below threshold {CORRECTNESS_THRESHOLD}. Reasoning: {reasoning}"
    )