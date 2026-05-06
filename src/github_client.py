"""Minimal GitHub Issues client."""

import os
from typing import Any, Dict, Iterable, List, Optional

import requests


class GitHubClient:
    """Simple GitHub Issues client using the public REST API."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
        if not self.token and os.getenv("GITHUB_PAT"):
            os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_PAT")
            self.token = os.getenv("GITHUB_PAT")

        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"

    def list_issues(
        self,
        repo: str,
        state: str = "open",
        per_page: int = 50,
        max_pages: Optional[int] = None,
        sort: str = "created",
        direction: str = "desc",
    ) -> Iterable[Dict[str, Any]]:
        """Yield issues for a repo.

        Args:
            repo: "owner/repo" (e.g., "huggingface/transformers").
            state: Open/closed/all.
            per_page: GitHub page size.
            max_pages: Maximum number of pages to fetch.
            sort: Sort field (created, updated, comments).
            direction: Sort direction (asc or desc).
        """

        owner, name = repo.split("/")
        url = f"{self.BASE_URL}/repos/{owner}/{name}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": 1,
            "sort": sort,
            "direction": direction,
        }

        page = 0
        while True:
            if max_pages is not None and page >= max_pages:
                break

            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            issues = resp.json()

            if not issues:
                break

            for issue in issues:
                # Skip pull requests since API returns PRs in issues endpoint.
                if "pull_request" in issue:
                    continue
                yield issue

            page += 1
            params["page"] += 1

    def get_issue(self, repo: str, issue_number: int) -> Dict[str, Any]:
        """Get a single issue by number."""

        owner, name = repo.split("/")
        url = f"{self.BASE_URL}/repos/{owner}/{name}/issues/{issue_number}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_latest_issue(
        self, repo: str, state: str = "open", per_page: int = 10, max_pages: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Return the latest non-PR issue for the repo.

        The GitHub Issues API returns pull requests mixed with issues, so we may need to
        walk a few pages to find the most recent actual issue.
        """

        # Try a few pages (few API calls) to find the first non-PR issue.
        for issue in self.list_issues(repo, state=state, per_page=per_page, max_pages=max_pages):
            return issue
        return None

    def search_issues(self, repo: str, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """Search issues and pull requests in the repository."""

        owner, name = repo.split("/")
        q = f"repo:{owner}/{name} {query} in:title,body"
        url = f"{self.BASE_URL}/search/issues"
        params = {
            "q": q,
            "per_page": per_page,
        }

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        return body.get("items", [])

    def list_contributors(self, repo: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """Get top contributors for the repository."""

        owner, name = repo.split("/")
        url = f"{self.BASE_URL}/repos/{owner}/{name}/contributors"
        params = {
            "per_page": per_page,
        }

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

