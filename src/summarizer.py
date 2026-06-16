"""Convert raw GitHub issue JSON into a stable internal structure."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "are",
    "before",
    "bug",
    "cannot",
    "does",
    "error",
    "fail",
    "fails",
    "failure",
    "for",
    "from",
    "github",
    "have",
    "how",
    "into",
    "issue",
    "not",
    "question",
    "the",
    "this",
    "using",
    "when",
    "with",
}


def _login(value: Optional[Dict[str, Any]]) -> Optional[str]:
    return (value or {}).get("login")


def _labels(issue: Dict[str, Any]) -> List[str]:
    labels = issue.get("labels", [])
    return [lbl.get("name") for lbl in labels if isinstance(lbl, dict) and lbl.get("name")]


def _assignees(issue: Dict[str, Any]) -> List[str]:
    return [
        assignee.get("login")
        for assignee in issue.get("assignees", [])
        if isinstance(assignee, dict) and assignee.get("login")
    ]


def _truncate(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 15].rstrip() + "\n...[truncated]"


def summarize_comment(comment: Dict[str, Any]) -> Dict[str, Any]:
    """Return the comment fields that matter for triage."""

    return {
        "id": comment.get("id"),
        "user": _login(comment.get("user")),
        "body": comment.get("body") or "",
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "url": comment.get("html_url") or comment.get("url"),
        "author_association": comment.get("author_association"),
    }


def summarize_related_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact metadata for a related issue search result."""

    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "state": issue.get("state"),
        "url": issue.get("html_url") or issue.get("url"),
        "labels": _labels(issue),
        "comments_count": issue.get("comments", 0),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
    }


def summarize_contributor(contributor: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact contributor metadata."""

    return {
        "login": contributor.get("login"),
        "contributions": contributor.get("contributions"),
        "url": contributor.get("html_url"),
    }


def summarize_repo_info(repo_info: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact repository metadata."""

    return {
        "full_name": repo_info.get("full_name"),
        "description": repo_info.get("description"),
        "language": repo_info.get("language"),
        "default_branch": repo_info.get("default_branch"),
        "open_issues_count": repo_info.get("open_issues_count"),
        "stars": repo_info.get("stargazers_count"),
        "url": repo_info.get("html_url"),
    }


def summarize_issue(
    issue: Dict[str, Any],
    comments: Optional[Iterable[Dict[str, Any]]] = None,
    related_issues: Optional[Iterable[Dict[str, Any]]] = None,
    contributors: Optional[Iterable[Dict[str, Any]]] = None,
    repo_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured representation of a GitHub issue."""

    summarized_comments = [summarize_comment(comment) for comment in comments or []]
    summarized_related = [
        summarize_related_issue(related)
        for related in related_issues or []
        if related.get("number") != issue.get("number")
    ]
    summarized_contributors = [
        summarize_contributor(contributor) for contributor in contributors or []
    ]

    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "url": issue.get("html_url") or issue.get("url"),
        "api_url": issue.get("url"),
        "state": issue.get("state"),
        "labels": _labels(issue),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "user": _login(issue.get("user")),
        "author_association": issue.get("author_association"),
        "comments_count": issue.get("comments", len(summarized_comments)),
        "assignees": _assignees(issue),
        "milestone": (issue.get("milestone") or {}).get("title"),
        "comments": summarized_comments,
        "related_issues": summarized_related,
        "contributors": summarized_contributors,
        "repo": summarize_repo_info(repo_info) if repo_info else {},
    }


def build_related_issue_query(issue: Dict[str, Any], max_terms: int = 6) -> str:
    """Build a concise search query for similar GitHub issues."""

    labels = issue.get("labels") or []
    source = " ".join([issue.get("title") or "", " ".join(labels)])
    tokens: List[str] = []

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", source):
        normalized = token.lower().strip(".-_")
        if normalized in STOPWORDS or normalized.isdigit():
            continue
        if normalized not in tokens:
            tokens.append(normalized)
        if len(tokens) >= max_terms:
            break

    if tokens:
        return " ".join(tokens)

    title = (issue.get("title") or "").strip()
    return title[:80]


def format_issue_text(
    issue: Dict[str, Any],
    max_body_chars: int = 8000,
    max_comments: int = 5,
    max_comment_chars: int = 1200,
    max_related: int = 5,
) -> str:
    """Format issue content into a single prompt text for the LM."""

    parts: List[str] = []
    title = issue.get("title")
    body = issue.get("body")
    repo = issue.get("repo") or {}

    if repo:
        repo_bits = [
            repo.get("full_name"),
            f"language={repo.get('language')}" if repo.get("language") else None,
            f"default_branch={repo.get('default_branch')}" if repo.get("default_branch") else None,
            (
                f"open_issues={repo.get('open_issues_count')}"
                if repo.get("open_issues_count") is not None
                else None
            ),
        ]
        parts.append("REPOSITORY: " + " | ".join(bit for bit in repo_bits if bit))

    metadata = [
        f"number=#{issue.get('number')}" if issue.get("number") is not None else None,
        f"state={issue.get('state')}" if issue.get("state") else None,
        f"author={issue.get('user')}" if issue.get("user") else None,
        (
            f"author_association={issue.get('author_association')}"
            if issue.get("author_association")
            else None
        ),
        f"created_at={issue.get('created_at')}" if issue.get("created_at") else None,
        f"updated_at={issue.get('updated_at')}" if issue.get("updated_at") else None,
        f"comments={issue.get('comments_count')}" if issue.get("comments_count") is not None else None,
    ]
    if any(metadata):
        parts.append("ISSUE METADATA: " + " | ".join(bit for bit in metadata if bit))

    if title:
        parts.append(f"TITLE: {title.strip()}")
    if body:
        parts.append("BODY:\n" + _truncate(body, max_body_chars))

    labels = issue.get("labels")
    if labels:
        parts.append("LABELS: " + ", ".join(labels))

    assignees = issue.get("assignees")
    if assignees:
        parts.append("ASSIGNEES: " + ", ".join(assignees))

    comments = issue.get("comments") or []
    if comments and max_comments > 0:
        comment_lines = []
        for idx, comment in enumerate(comments[:max_comments], start=1):
            user = comment.get("user") or "unknown"
            created_at = comment.get("created_at") or "unknown date"
            body_text = _truncate(comment.get("body"), max_comment_chars)
            comment_lines.append(f"{idx}. {user} at {created_at}:\n{body_text}")
        parts.append("ISSUE COMMENTS:\n" + "\n\n".join(comment_lines))

    related = issue.get("related_issues") or []
    if related and max_related > 0:
        related_lines = []
        for item in related[:max_related]:
            labels_text = ", ".join(item.get("labels") or [])
            suffix = f" [{labels_text}]" if labels_text else ""
            related_lines.append(
                f"- #{item.get('number')} {item.get('state')}: {item.get('title')}{suffix}"
            )
        parts.append("POTENTIALLY RELATED ISSUES:\n" + "\n".join(related_lines))

    contributors = issue.get("contributors") or []
    if contributors:
        names = [
            f"{contributor.get('login')} ({contributor.get('contributions')})"
            for contributor in contributors
            if contributor.get("login")
        ]
        if names:
            parts.append("TOP CONTRIBUTORS: " + ", ".join(names))

    return "\n\n".join(parts).strip()
