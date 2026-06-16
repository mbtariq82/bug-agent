import unittest

from src.issue_advisor import IssueAdvisor


class IssueAdvisorTests(unittest.TestCase):
    def test_no_model_mode_returns_deterministic_triage(self):
        advisor = IssueAdvisor(use_model=False)

        response = advisor.advise(
            "\n".join(
                [
                    "TITLE: Tokenizer crashes on empty input",
                    "LABELS: bug, tokenizers",
                    "POTENTIALLY RELATED ISSUES:",
                    "- #9 closed: Tokenizer fails with blank strings",
                ]
            ),
            issue_number=12,
        )

        self.assertIn("Issue #12: Tokenizer crashes on empty input", response)
        self.assertIn("Model-backed analysis is unavailable", response)
        self.assertIn("minimal reproduction steps", response)
        self.assertIn("Compare against the related issues", response)


if __name__ == "__main__":
    unittest.main()
