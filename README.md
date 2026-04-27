# Bug Agent
An autonomous agent to help maintainers research issues/bugs. Potential workflows include the following:
- searching through existing issues/PRs
- recreating a bug
- writing a comment (see github.com/huggingface/transformers/issues/44485 → references to `vllm` and `sglang`)
- summary of a PR for a specific maintainer

Pipeline:
1. **GitHub API**
2. **Issue Summarizer** / **PR Summarizer**
4. **LM advisor**
5. **SurrealDB for persistent memory**


## Quick Start
(Optional) Change the model used by the advisor. Default is `HuggingFaceTB/SmolLM-1.7B`:

```bash
export MODEL_NAME=HuggingFaceTB/SmolLM-1.7B
# Windows PowerShell
$env:MODEL_NAME = 'HuggingFaceTB/SmolLM-1.7B'
```

Run the agent (defaults to the latest open Transformers issue):

```bash
python -m src.main
```

- To process a specific issue:

```bash
python -m src.main --issue 44593
```

