import unittest

from src.github_client import GitHubClient, split_repo


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        return FakeResponse(self.responses.pop(0))


class GitHubClientTests(unittest.TestCase):
    def test_split_repo_validates_owner_name(self):
        self.assertEqual(split_repo("owner/name"), ("owner", "name"))

        with self.assertRaises(ValueError):
            split_repo("owner")

        with self.assertRaises(ValueError):
            split_repo("owner/")

    def test_search_issues_adds_issue_filter_and_removes_pull_requests(self):
        client = GitHubClient(token="token")
        client.session = FakeSession(
            [
                {
                    "items": [
                        {"number": 1, "title": "real issue"},
                        {
                            "number": 2,
                            "title": "pull request",
                            "pull_request": {"url": "https://api.github.com/pr/2"},
                        },
                    ]
                }
            ]
        )

        results = client.search_issues("owner/name", "tokenizer crash", per_page=5)

        self.assertEqual([result["number"] for result in results], [1])
        params = client.session.calls[0]["params"]
        self.assertIn("repo:owner/name is:issue tokenizer crash in:title,body", params["q"])
        self.assertEqual(params["per_page"], 5)

    def test_get_issue_comments_paginates(self):
        client = GitHubClient(token="token")
        client.session = FakeSession(
            [
                [{"id": 1, "body": "first"}],
                [{"id": 2, "body": "second"}],
            ]
        )

        comments = client.get_issue_comments(
            "owner/name",
            issue_number=10,
            per_page=1,
            max_pages=2,
        )

        self.assertEqual([comment["id"] for comment in comments], [1, 2])
        self.assertEqual(len(client.session.calls), 2)
        self.assertEqual(client.session.calls[0]["params"]["page"], 1)
        self.assertEqual(client.session.calls[1]["params"]["page"], 2)


if __name__ == "__main__":
    unittest.main()
