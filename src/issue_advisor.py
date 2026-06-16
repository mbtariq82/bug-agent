"""
Issue advisor using TransformersModel for analyzing GitHub issues.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = (
        "You are an assistant that analyzes GitHub issues. Provide concise triage "
        "guidance with likely root causes, missing diagnostic details, reproduction "
        "steps, related-issue clues, and concrete next actions."
    )

    def __init__(self, model_name: Optional[str] = None, use_model: bool = True):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        self.use_model = use_model
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HF_API_KEY")
        )

        if self.hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token

        if not self.use_model:
            self.tokenizer = None
            self.model = None
            self.fallback_reason = "model disabled"
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.fallback_reason = None
        except Exception as e:
            LOG.error("Model loading failed: %s", str(e))
            LOG.warning("Falling back to a simple advice generator (no ML model)")
            self.tokenizer = None
            self.model = None
            self.fallback_reason = str(e)
            return
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Generate advice for the issue."""
        if self.model is None or self.tokenizer is None:
            return self._fallback_advice(issue_text, issue_number, self.fallback_reason)
        
        prompt = f"{self.SYSTEM_PROMPT}\n\nIssue #{issue_number}:\n{issue_text}\n\nAdvice:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=256,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if prompt in response:
            response = response.split(prompt, 1)[-1].strip()
        
        return response

    def _fallback_advice(
        self,
        issue_text: str,
        issue_number: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> str:
        """Generate basic advice without an ML model."""

        lower_text = issue_text.lower()
        title_match = re.search(r"^TITLE:\s*(.+)$", issue_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Unknown issue"
        labels_match = re.search(r"^LABELS:\s*(.+)$", issue_text, re.MULTILINE)
        labels = labels_match.group(1).strip() if labels_match else "none"

        missing_details = []
        if not any(term in lower_text for term in ("repro", "reproduction", "steps to reproduce")):
            missing_details.append("minimal reproduction steps")
        if not any(term in lower_text for term in ("traceback", "stack trace", "exception", "error:", "logs")):
            missing_details.append("complete error output or logs")
        if not any(term in lower_text for term in ("version", "environment", "python", "node", "os:", "platform")):
            missing_details.append("environment and dependency versions")
        if not any(term in lower_text for term in ("expected", "actual")):
            missing_details.append("expected vs actual behavior")

        context_notes = []
        if "issue comments:" in lower_text:
            context_notes.append("Review the discussion for maintainer requests and attempted workarounds.")
        if "potentially related issues:" in lower_text:
            context_notes.append("Compare against the related issues before treating this as novel.")
        if "assignees:" in lower_text:
            context_notes.append("An assignee is already present, so next steps should preserve existing ownership.")

        reason_text = f" ({reason})" if reason else ""
        lines = [
            f"Issue #{issue_number}: {title}",
            "",
            f"Model-backed analysis is unavailable{reason_text}; using deterministic triage.",
            "",
            f"Labels: {labels}",
            "",
            "Likely next actions:",
            "1. Reproduce locally with the reporter's exact inputs and versions.",
            "2. Search the implicated area for recent changes matching the failure mode.",
            "3. Add or update a regression test once the trigger is isolated.",
            "4. Ask the reporter for any missing diagnostic details before implementation.",
        ]

        if missing_details:
            lines.extend(["", "Missing details to request:"])
            lines.extend(f"- {detail}" for detail in missing_details)

        if context_notes:
            lines.extend(["", "Context notes:"])
            lines.extend(f"- {note}" for note in context_notes)

        return "\n".join(lines)

