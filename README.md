# LLM Eval Suite

## Requirements

### System
* Python3
* pip 25.2

### API Keys
* OpenAI API Key
* DeepEval API Key

## Set Up

### Python Virtual Environment and installing requirements

```python3 -m venv <myvenv>```

```source <myenv>/bin/activate```

```pip install -r requirements.txt```


### Setting up Ollama

* Install Ollama by downloading from 
  * [MAC](https://ollama.com/download/mac)
  * [LINUX](https://ollama.com/download/linux)
  * [Windows](https://ollama.com/download/windows)

The following instructions are for mac/linux. Feel free to contribute for Windows.

1. Run Ollama<br>
``` ollama serve ```
2. Pull a model<br>
```ollama pull llama3.1:8b```
3. List models<br>
```ollama list```

You should see llama3.1:8b listed. This is currently what the test suite uses as the model under test for the LLM As A Judge Test Suite

### API Keys

To run the test suite, you will need the following API keys:

* OpenAI Key
* DeepEval Key

## How to run test

### DeepEvals

#### LLM AS A JUDGE

```pytest -q tests/llm-as-a-judge/correctness/test_deepeval.py```

#### How To View Results
By default, results are written to logs/deepeval_runs.jsonl (configurable via EVAL_LOG_FILE in .env). 
Open view_evals_results.html in a browser and use the file picker to load your JSONL results.