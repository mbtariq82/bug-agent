"""Entry point for the Bug Agent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import requests

from .issue_advisor import IssueAdvisor
from .github_client import GitHubClient
from .summarizer import build_related_issue_query, summarize_issue, format_issue_text


LOG = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate with the Hugging Face model.",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading a Hugging Face model and use deterministic triage.",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=0,
        help="Number of issue comments to include in the analysis prompt.",
    )
    parser.add_argument(
        "--related",
        type=int,
        default=0,
        help="Number of potentially related issues to include from GitHub search.",
    )
    parser.add_argument(
        "--contributors",
        type=int,
        default=0,
        help="Number of top repository contributors to include as context.",
    )
    parser.add_argument(
        "--repo-context",
        action="store_true",
        help="Include repository metadata such as language and default branch.",
    )
    parser.add_argument(
        "--memory-file",
        default="bug_agent_memory.json",
        help="Path to the JSON memory file. Defaults to bug_agent_memory.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured analysis as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def store_analysis(
    repo: str,
    issue: Dict[str, Any],
    response: str,
    memory_file: str = "bug_agent_memory.json",
) -> None:
    """Store the analysis in a JSON file for persistent memory."""
    data = {}
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            LOG.warning("Memory file %s is not valid JSON; starting fresh", memory_file)
    
    issue_number = issue.get("number")
    key = f"{repo}#{issue_number}"
    data[key] = {
        "repo": repo,
        "issue": issue,
        "response": response,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_dotenv(dotenv_name: str = ".env") -> Optional[str]:
    """Locate a .env file by searching upward from the current working directory."""
    current_dir = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(current_dir, dotenv_name)
        if os.path.exists(candidate):
            return candidate
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, dotenv_name)
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.normpath(os.path.join(script_dir, "..", dotenv_name))
    if os.path.exists(candidate):
        return candidate

    return None


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into the environment."""
    if not os.path.exists(dotenv_path):
        dotenv_path = find_dotenv(dotenv_path)
    if not dotenv_path or not os.path.exists(dotenv_path):
        return

    LOG.debug("Loading .env from %s", dotenv_path)
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def load_colab_secrets() -> None:
    """Load secrets from Google Colab's Secrets sidebar into the environment."""
    try:
        from google.colab import userdata
        secret_names = ["HF_TOKEN", "HF_API_KEY", "HUGGINGFACE_HUB_TOKEN", "GITHUB_TOKEN", "GITHUB_PAT"]
        for secret_name in secret_names:
            if secret_name not in os.environ:
                try:
                    value = userdata.get(secret_name)
                    if value:
                        os.environ[secret_name] = value
                        LOG.debug("Loaded secret %s from Colab", secret_name)
                except (userdata.NotebookAccessError, AttributeError):
                    pass
    except (ImportError, AttributeError):
        pass


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    load_colab_secrets()
    args = parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    LOG.debug(
        "Environment variables: HF_TOKEN=%s HF_API_KEY=%s HUGGINGFACE_HUB_TOKEN=%s GITHUB_TOKEN=%s GITHUB_PAT=%s MODEL_NAME=%s",
        bool(os.getenv("HF_TOKEN")),
        bool(os.getenv("HF_API_KEY")),
        bool(os.getenv("HUGGINGFACE_HUB_TOKEN")),
        bool(os.getenv("GITHUB_TOKEN")),
        bool(os.getenv("GITHUB_PAT")),
        os.getenv("MODEL_NAME"),
    )

    client = GitHubClient()

    try:
        issue = None
        if args.issue is not None:
            LOG.info("Fetching issue #%d", args.issue)
            issue = client.get_issue(args.repo, args.issue)
        else:
            LOG.info("Fetching latest open issue")
            issue = client.get_latest_issue(args.repo)
    except (requests.RequestException, ValueError) as e:
        LOG.error("Failed to fetch issue: %s", str(e))
        return 1

    if issue is None:
        LOG.warning("No issue found to analyze.")
        return 0

    LOG.info("Fetched issue #%s: %s", issue.get("number"), issue.get("title"))

    number = issue.get("number")
    comments = []
    related_issues = []
    contributors = []
    repo_info = None

    try:
        if args.comments > 0 and number is not None:
            pages = max(1, math.ceil(args.comments / 100))
            comments = client.get_issue_comments(
                args.repo,
                number,
                per_page=min(args.comments, 100),
                max_pages=pages,
            )[: args.comments]

        base_structured = summarize_issue(issue)
        if args.related > 0:
            query = build_related_issue_query(base_structured)
            if query:
                related_issues = [
                    item
                    for item in client.search_issues(
                        args.repo,
                        query,
                        per_page=args.related + 1,
                    )
                    if item.get("number") != number
                ][: args.related]

        if args.contributors > 0:
            contributors = client.list_contributors(args.repo, per_page=args.contributors)

        if args.repo_context:
            repo_info = client.get_repo_info(args.repo)
    except (requests.RequestException, ValueError) as e:
        LOG.error("Failed to fetch enrichment context: %s", str(e))
        return 1

    structured = summarize_issue(
        issue,
        comments=comments,
        related_issues=related_issues,
        contributors=contributors,
        repo_info=repo_info,
    )
    number = structured.get("number")

    prompt = format_issue_text(structured)
    if not prompt:
        LOG.warning("Issue #%s has no text to analyze", number)
        return 0

    try:
        if args.no_model:
            LOG.info("Generating deterministic triage (--no-model)")
        else:
            model_name = args.model_name or os.getenv("MODEL_NAME") or IssueAdvisor.DEFAULT_MODEL
            LOG.info(
                "Loading advisor model %s; use --no-model for a fast deterministic run",
                model_name,
            )
        advisor = IssueAdvisor(
            model_name=args.model_name,
            use_model=not args.no_model,
            max_new_tokens=args.max_new_tokens,
        )
        LOG.info("Generating analysis")
        response = advisor.advise(prompt, number)
        LOG.info("Analysis generated")
    except Exception as e:
        LOG.error("Advisor failed: %s", str(e))
        return 1

    store_analysis(args.repo, structured, response, memory_file=args.memory_file)

    LOG.info("#%s %s => %s", number, structured.get("title"), response.replace("\n", " ")[:80])

    if args.json:
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "issue": structured,
                    "response": response,
                },
                indent=2,
            )
        )
        return 0

    print("---")
    print(f"#{number} {structured.get('title')}")
    print(f"URL: {structured.get('url')}")
    if structured.get("comments"):
        print(f"Included comments: {len(structured.get('comments'))}")
    if structured.get("related_issues"):
        print(f"Related issues: {len(structured.get('related_issues'))}")
    print("Response:")
    print(response)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
