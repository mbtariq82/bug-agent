# Bug Agent

A simple GitHub issue analysis tool that fetches GitHub issues, summarizes the issue content, and asks a local language model for advice or next steps.

## What this agent does

- Fetches an issue from a GitHub repository
- Converts issue metadata into a structured internal format
- Optionally enriches analysis with comments, related issues, repository metadata, and top contributors
- Builds a prompt containing issue details and selected context
- Sends the prompt to a local Hugging Face causal LM for advice, or uses deterministic triage with `--no-model`
- Saves analysis results to a local JSON file for later reference

## Current implementation

The current code path includes:

1. `src/main.py`
   - parses command-line options
   - loads the GitHub client and the issue advisor
   - fetches either a specific issue or the latest open issue
   - optionally fetches comments, related issues, contributors, and repo metadata
   - summarizes and formats the issue context
   - calls the advisor to generate natural-language guidance
   - writes the result into `bug_agent_memory.json`

2. `src/github_client.py`
   - uses the GitHub REST API to fetch issues
   - supports retrieving one issue, the latest open issue, issue comments, issue search, contributors, and repository metadata
   - skips pull requests when enumerating issues

3. `src/summarizer.py`
   - converts raw GitHub issue JSON into a structured object
   - builds a related-issue search query from issue-specific terms
   - formats issue context into prompt text for the LM

4. `src/issue_advisor.py`
   - loads a Hugging Face causal LM model and tokenizer
   - constructs a prompt with a system instruction and issue details
   - generates advice text from the model output
   - falls back to deterministic triage when the model is disabled or unavailable

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

Run a fast deterministic analysis without loading a Hugging Face model:

```bash
python -m src.main --no-model
```

If model-backed analysis is slow on CPU, limit generation length:

```bash
python -m src.main --max-new-tokens 80
```

Run the agent for a specific issue number:

```bash
python -m src.main --issue 44593
```

Run the agent for a different repository:

```bash
python -m src.main --repo owner/repo
```

Include richer GitHub context:

```bash
python -m src.main --repo owner/repo --issue 123 --comments 5 --related 5 --contributors 3 --repo-context
```

Print the full structured result as JSON:

```bash
python -m src.main --issue 44593 --comments 3 --related 3 --no-model --json
```

Run the test suite:

```bash
python -m unittest discover
```

## CLI options

- `--comments N`: include up to `N` issue comments.
- `--related N`: search for up to `N` potentially related issues.
- `--contributors N`: include the top `N` repository contributors.
- `--repo-context`: include repository metadata such as language and default branch.
- `--no-model`: skip Hugging Face model loading and use deterministic triage.
- `--max-new-tokens N`: limit Hugging Face generation length. Lower values return faster.
- `--memory-file PATH`: write persistent analysis memory to a custom JSON file.
- `--json`: print the structured issue context and response as JSON.
- `--verbose`: enable debug logging.

## Output

The agent prints the selected issue URL and analysis response, and stores the structured issue context plus response in `bug_agent_memory.json`.
