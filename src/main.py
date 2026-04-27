"""Entry point for the Bug Agent."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

from .issue_advisor import IssueAdvisor
from .github_client import GitHubClient
from .summarizer import summarize_issue, format_issue_text


LOG = logging.getLogger(__name__)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Bug Agent for a GitHub repo."
    )
    parser.add_argument(
        "--repo",
        default="huggingface/transformers",
        help="Repository to scan (owner/repo). Defaults to huggingface/transformers.",
    )
    parser.add_argument(
        "--issue",
        type=int,
        help="Specific issue number to analyze (defaults to latest open issue).",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=None,
        help="Hugging Face model name for the advisor (overrides MODEL_NAME env var).",
    )
    return parser.parse_args(argv)


def store_analysis(repo: str, issue_number: int, title: str, response: str):
    """Store the analysis in a JSON file for persistent memory."""
    data_file = "bug_agent_memory.json"
    data = {}
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
    
    key = f"{repo}#{issue_number}"
    data[key] = {
        "title": title,
        "response": response,
    }
    
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.DEBUG,
    )

    client = GitHubClient()
    advisor = IssueAdvisor(model_name=args.model_name)

    issue = None
    if args.issue is not None:
        LOG.info("Fetching issue #%d", args.issue)
        issue = client.get_issue(args.repo, args.issue)
    else:
        LOG.info("Fetching latest open issue")
        issue = client.get_latest_issue(args.repo)

    if issue is None:
        LOG.warning("No issue found to analyze.")
        return 0
    
    structured = summarize_issue(issue)
    number = structured.get("number")

    prompt = format_issue_text(structured)
    if not prompt:
        LOG.warning("Issue #%s has no text to analyze", number)
        return 0

    try:
        response = advisor.advise(prompt, number)
    except Exception as e:
        LOG.error("Advisor failed: %s", str(e))
        return 1

    # Store in persistent memory
    store_analysis(args.repo, number, structured.get("title"), response)

    # Log and print
    LOG.info("#%s %s => %s", number, structured.get("title"), response.replace("\n", " ")[:80])

    print("---")
    print(f"#{number} {structured.get('title')}")
    print(f"URL: {structured.get('url')}")
    print("Response:")
    print(response)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
