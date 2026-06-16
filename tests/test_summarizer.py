import unittest

from src.summarizer import build_related_issue_query, format_issue_text, summarize_issue


class SummarizerTests(unittest.TestCase):
    def test_summarize_issue_includes_enrichment(self):
        issue = {
            "number": 12,
            "title": "Tokenizer crashes on empty input",
            "body": "Calling tokenize('') raises ValueError.",
            "html_url": "https://github.com/example/project/issues/12",
            "state": "open",
            "labels": [{"name": "bug"}, {"name": "tokenizers"}],
            "user": {"login": "reporter"},
            "comments": 1,
            "assignees": [{"login": "maintainer"}],
        }
        comments = [
            {
                "id": 1,
                "body": "I can reproduce on Python 3.12.",
                "user": {"login": "maintainer"},
                "created_at": "2026-06-16T10:00:00Z",
            }
        ]
        related = [
            {"number": 12, "title": "same issue"},
            {
                "number": 9,
                "title": "Tokenizer fails with blank strings",
                "state": "closed",
                "labels": [{"name": "tokenizers"}],
            },
        ]
        contributors = [{"login": "core-dev", "contributions": 42}]
        repo_info = {
            "full_name": "example/project",
            "language": "Python",
            "default_branch": "main",
            "open_issues_count": 7,
        }

        structured = summarize_issue(
            issue,
            comments=comments,
            related_issues=related,
            contributors=contributors,
            repo_info=repo_info,
        )

        self.assertEqual(structured["number"], 12)
        self.assertEqual(structured["labels"], ["bug", "tokenizers"])
        self.assertEqual(structured["assignees"], ["maintainer"])
        self.assertEqual(len(structured["comments"]), 1)
        self.assertEqual(len(structured["related_issues"]), 1)
        self.assertEqual(structured["repo"]["language"], "Python")

        prompt = format_issue_text(structured)
        self.assertIn("REPOSITORY: example/project", prompt)
        self.assertIn("ISSUE COMMENTS", prompt)
        self.assertIn("POTENTIALLY RELATED ISSUES", prompt)
        self.assertIn("TOP CONTRIBUTORS: core-dev (42)", prompt)

    def test_build_related_issue_query_uses_specific_terms(self):
        issue = {
            "title": "Bug: tokenizer crashes when loading safetensors shard",
            "labels": ["bug", "tokenizers", "model-loading"],
        }

        query = build_related_issue_query(issue)

        self.assertIn("tokenizer", query)
        self.assertIn("safetensors", query)
        self.assertNotIn("bug", query.split())


if __name__ == "__main__":
    unittest.main()
