# Bug Agent

A simple GitHub issue analysis tool that fetches GitHub issues, summarizes the issue content, and asks a local language model for advice or next steps.

## What this agent does

- Fetches an issue from a GitHub repository
- Converts issue metadata into a structured internal format
- Builds a prompt containing the issue title, body, and labels
- Sends the prompt to a local Hugging Face causal LM for advice
- Saves analysis results to a local JSON file for later reference

## Current implementation

The current code path includes:

1. `src/main.py`
   - parses command-line options
   - loads the GitHub client and the issue advisor
   - fetches either a specific issue or the latest open issue
   - summarizes and formats the issue text
   - calls the advisor to generate natural-language guidance
   - writes the result into `bug_agent_memory.json`

2. `src/github_client.py`
   - uses the GitHub REST API to fetch issues
   - supports retrieving one issue, the latest open issue, issue search, and contributor listing
   - skips pull requests when enumerating issues

3. `src/summarizer.py`
   - converts raw GitHub issue JSON into a structured object
   - formats title, body, and labels into prompt text for the LM

4. `src/issue_advisor.py`
   - loads a Hugging Face causal LM model and tokenizer
   - constructs a prompt with a system instruction and issue details
   - generates advice text from the model output

## Requirements

- Python 3.10+
- `requests`
- `transformers`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Optionally set a custom model name:

```bash
export MODEL_NAME=HuggingFaceTB/SmolLM-1.7B
```

Run the agent for the default repository and latest open issue:

```bash
python -m src.main
```

Run the agent for a specific issue number:

```bash
python -m src.main --issue 44593
```

Run the agent for a different repository:

```bash
python -m src.main --repo owner/repo
```

## Output

The agent prints the selected issue URL and the model's advice, and also stores the result in `bug_agent_memory.json`.

## Notes

- The repository currently does not include SurrealDB or any external persistent database integration.
- If `GITHUB_TOKEN` is set, the GitHub client will authenticate requests with it.
- If `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY` is set, the Hugging Face model download will use authenticated requests and higher rate limits.
- A `.env` file in the repo root is automatically loaded by `src/main.py` when the agent starts.
- The default model is `HuggingFaceTB/SmolLM-1.7B`, but you can override it with `MODEL_NAME` or `--model`.
