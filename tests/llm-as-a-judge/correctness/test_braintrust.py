import os
from pathlib import Path
import braintrust as bt
from openai import OpenAI
import json
import pytest

# Load .env locally only; avoid overriding secrets in CI
if os.getenv("CI") != "true" and os.getenv("GITHUB_ACTIONS") != "true":
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

# --- Config ---
DATA_FILE = os.getenv("DATASET_PATH", "data_sets/sample_automotive_supply_chain.jsonl")
MODEL_RUNNER = os.getenv("MODEL_RUNNER", "openai")  # ollama | http | noop | openai
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_ENDPOINT = os.getenv("APP_ENDPOINT", "http://localhost:8000/generate")
CORRECTNESS_THRESHOLD = float(os.getenv("CORRECTNESS_THRESHOLD", "0.5"))

USE_PRECOMPUTED_OUTPUTS = os.getenv("USE_PRECOMPUTED_OUTPUTS", "false").lower() in ("1", "true", "yes")

# Use JUDGEMENT_MODEL from env (for consistency with other evals) or fallback to OPENAI_JUDGE_MODEL
JUDGEMENT_MODEL = os.getenv("JUDGEMENT_MODEL") or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CI/automation: configure project + API key without braintrust init
BT_API_KEY = os.getenv("BRAINTRUST_API_KEY")
BT_PROJECT = os.getenv("BRAINTRUST_PROJECT", "automotive-supply-chain")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "braintrust_eval.jsonl"


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
        if client is None:
            raise RuntimeError("MODEL_RUNNER=openai but OPENAI_API_KEY is not set")
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


def get_dataset_or_model_output(record: dict) -> str:
    """
    If USE_PRECOMPUTED_OUTPUTS is enabled and the dataset record has an 'output' (or 'actual'),
    return that. Otherwise, fall back to generating it with the configured model runner.
    """
    if USE_PRECOMPUTED_OUTPUTS:
        if "output" in record and record["output"]:
            return str(record["output"]).strip()
        if "actual" in record and record["actual"]:
            return str(record["actual"]).strip()
        raise ValueError("USE_PRECOMPUTED_OUTPUTS set but no 'output' field found in dataset record")
    # fallback: call model
    return get_model_output(record["input"])


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

        # Add difficulty from metadata if present
        metadata = obj.get("metadata", {})
        if isinstance(metadata, dict):
            if "difficulty" in metadata:
                braintrust_record["difficulty"] = metadata["difficulty"]
            # Optionally carry all metadata fields up
            braintrust_record.update(metadata)

        # Add any top-level fields besides input/ideal/metadata
        for key, value in obj.items():
            if key not in ["input", "ideal", "metadata"]:
                braintrust_record[key] = value

        records.append(braintrust_record)

print(f"Loaded {len(records)} valid records from dataset")

if not records:
    pytest.skip(f"No valid records loaded from dataset: {data_path.resolve()}", allow_module_level=True)


# --- Braintrust Evaluation Function ---
def run_braintrust_eval():
    """Run evaluation using Braintrust framework and send results to dashboard"""
    
    # Debug: Check authentication
    print(f"Braintrust API Key set: {bool(BT_API_KEY)}")
    print(f"Project name: {BT_PROJECT}")
    
    # Configure Braintrust with explicit error handling
    if not BT_API_KEY:
        print("✗ No BRAINTRUST_API_KEY found. Please set this environment variable.")
        print("You can get an API key from: https://www.braintrust.dev/app/settings?subroute=api-keys")
        return None
    
    try:
        bt.login(api_key=BT_API_KEY)
        print("✓ Successfully authenticated with Braintrust")
    except Exception as e:
        print(f"✗ Failed to authenticate with Braintrust: {e}")
        return None
    
    # Create or get project with error handling
    try:
        project = bt.init_project(name=BT_PROJECT)
        print(f"✓ Project initialized: {BT_PROJECT}")
    except Exception as e:
        print(f"✗ Failed to initialize project '{BT_PROJECT}': {e}")
        return None
    
    def task_function(input_text):
        """Task function that generates output for given input"""
        try:
            # For Braintrust, we need to create a record-like dict to use our existing function
            record = {"input": input_text}
            output = get_dataset_or_model_output(record)
            print(f"Generated output for: {input_text[:50]}... -> {output[:50]}...")
            return output
        except Exception as e:
            print(f"Error in task_function: {e}")
            return f"Error: {e}"
    
    # Prepare data with validation
    eval_data = []
    for i, record in enumerate(records):
        if not record.get("input") or not record.get("expected"):
            print(f"Skipping record {i}: missing input or expected")
            continue
        eval_data.append({
            "input": record["input"],
            "expected": record["expected"],
            "metadata": {
                "difficulty": record.get("difficulty", ""),
                "id": record.get("id", i)
            }
        })
    
    print(f"Prepared {len(eval_data)} records for evaluation")
    
    if not eval_data:
        print("✗ No valid data to evaluate")
        return None
    
    # Run the evaluation with detailed error handling
    try:
        print("Starting Braintrust evaluation...")
        eval_result = bt.eval(
            name=f"Automotive Supply Chain Eval - {MODEL_RUNNER}",
            project=BT_PROJECT,
            data=eval_data,
            task=task_function,
            scores=[bt.scorer(correctness_eval, name="correctness")],
            experiment_name=f"{MODEL_RUNNER}_{OPENAI_MODEL if MODEL_RUNNER == 'openai' else OLLAMA_MODEL}",
            metadata={
                "model_runner": MODEL_RUNNER,
                "generator_model": OPENAI_MODEL if MODEL_RUNNER == "openai" else OLLAMA_MODEL,
                "judgement_model": JUDGEMENT_MODEL,
                "correctness_threshold": CORRECTNESS_THRESHOLD,
            }
        )
        print("✓ Braintrust evaluation completed successfully")
    except Exception as e:
        print(f"✗ Failed to run Braintrust evaluation: {e}")
        print("Make sure:")
        print("1. Your API key has the right permissions")
        print("2. The project exists or you have permission to create it")
        print("3. Your internet connection is stable")
        return None
    
    # Write results to log file
    print(f"\nWriting results to log file: {LOG_FILE}")
    for i, result in enumerate(eval_result.results):
        if result is None:
            continue
            
        # Get the original record for metadata
        original_record = records[i] if i < len(records) else {}
        
        # Extract score and metadata
        score = result.get("scores", {}).get("correctness", 0.0)
        reasoning = "No reasoning available"
        
        # Try to get reasoning from score metadata
        if isinstance(score, dict):
            reasoning = score.get("metadata", {}).get("reasoning", reasoning)
            score = score.get("score", 0.0)
        elif hasattr(result, "scores") and hasattr(result.scores, "correctness"):
            score_obj = result.scores.correctness
            if hasattr(score_obj, "metadata"):
                reasoning = score_obj.metadata.get("reasoning", reasoning)
            if hasattr(score_obj, "score"):
                score = score_obj.score
        
        log_entry = {
            "idx": original_record.get("id", i),
            "difficulty": original_record.get("difficulty", ""),
            "score": float(score) if score is not None else 0.0,
            "threshold": CORRECTNESS_THRESHOLD,
            "status": "PASS" if (score is not None and float(score) >= CORRECTNESS_THRESHOLD) else "FAIL",
            "generator_model": OPENAI_MODEL if MODEL_RUNNER == "openai" else OLLAMA_MODEL,
            "judgement_model": JUDGEMENT_MODEL,
            "input": result.get("input", original_record.get("input", "")),
            "expected": result.get("expected", original_record.get("expected", "")),
            "actual": result.get("output", ""),
            "judgement_reason": reasoning,
            "experiment_url": eval_result.experiment_url,
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    print(f"\nBraintrust evaluation completed!")
    print(f"Experiment URL: {eval_result.experiment_url}")
    print(f"Project URL: {eval_result.project_url}")
    print(f"Results logged to: {LOG_FILE}")
    
    # Calculate summary statistics
    scores = []
    for result in eval_result.results:
        if result and result.get("scores", {}).get("correctness") is not None:
            score = result["scores"]["correctness"]
            # Handle both numeric scores and dict scores
            if isinstance(score, dict):
                scores.append(score.get("score", 0.0))
            else:
                scores.append(float(score))
    
    if scores:
        avg_score = sum(scores) / len(scores)
        pass_rate = sum(1 for score in scores if score >= CORRECTNESS_THRESHOLD) / len(scores)
        print(f"Average correctness score: {avg_score:.3f}")
        print(f"Pass rate (>= {CORRECTNESS_THRESHOLD}): {pass_rate:.1%}")
        print(f"Total records processed: {len(scores)}")
    else:
        print("No valid scores found in results")
    
    return eval_result


# --- Individual pytest tests (for local development) ---
@pytest.mark.parametrize("record", records)
def test_individual_record(record):
    """Test each record individually (creates separate tests for each record)"""
    print(f"\n=== INDIVIDUAL RECORD TEST ===")
    print(f"Testing record: {record.get('input', '')[:100]}...")
    print(f"Expected: {record.get('expected', '')[:50]}...")
    
    # Get output (from dataset if configured, else run model)
    output = get_dataset_or_model_output(record)
    print(f"Model output: {output[:50]}...")

    # Evaluate correctness using the scorer
    result = correctness_eval(record, output)
    score = result["score"]
    reasoning = result.get("metadata", {}).get("reasoning", "No reasoning provided")

    log_entry = {
        "idx": record.get("id", len(records)),
        "difficulty": record.get("difficulty", ""),
        "score": score,
        "threshold": CORRECTNESS_THRESHOLD,
        "status": "PASS" if score >= CORRECTNESS_THRESHOLD else "FAIL",
        "generator_model": OPENAI_MODEL if MODEL_RUNNER == "openai" else OLLAMA_MODEL,
        "judgement_model": JUDGEMENT_MODEL,
        "input": record["input"],
        "expected": record["expected"],
        "actual": output,
        "judgement_reason": reasoning,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"Score: {score:.3f}, Threshold: {CORRECTNESS_THRESHOLD}")
    
    # Assert the score meets the threshold
    assert score >= CORRECTNESS_THRESHOLD, (
        f"Score {score:.3f} below threshold {CORRECTNESS_THRESHOLD}. Reasoning: {reasoning}"
    )


# --- Main execution ---
if __name__ == "__main__":
    """
    Run Braintrust evaluation when script is executed directly.
    This allows you to run: python script_name.py
    """
    try:
        eval_result = run_braintrust_eval()
    except Exception as e:
        print(f"Error running Braintrust evaluation: {e}")
        print("Make sure BRAINTRUST_API_KEY is set and you have access to the project.")
        raise