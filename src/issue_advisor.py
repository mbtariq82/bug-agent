"""
Issue advisor using TransformersModel for analyzing GitHub issues.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = "You are an assistant that analyzes GitHub issues and provides guidance on potential root causes, reproduction steps, and next actions."

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HF_API_KEY")
        )

        if self.hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except ImportError as e:
            LOG.error("PyTorch or model loading failed: %s", str(e))
            LOG.warning("Falling back to a simple advice generator (no ML model)")
            self.tokenizer = None
            self.model = None
            return
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Generate advice for the issue."""
        if self.model is None or self.tokenizer is None:
            return self._fallback_advice(issue_text, issue_number)
        
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

    def _fallback_advice(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Generate basic advice without an ML model."""
        lines = issue_text.split('\n')
        title = lines[0] if lines else "Unknown issue"
        
        advice = f"Issue #{issue_number}: {title}\n\n"
        advice += "Unable to load ML model (PyTorch not available). Here are basic troubleshooting steps:\n\n"
        advice += "1. **Check the error logs**: Look for the exact error message and stack trace.\n"
        advice += "2. **Isolate the problem**: Create a minimal reproduction script that triggers the issue.\n"
        advice += "3. **Check dependencies**: Verify all package versions are compatible.\n"
        advice += "4. **Check for known issues**: Search GitHub issues for similar problems.\n"
        advice += "5. **Provide environment details**: Include OS, Python version, and library versions.\n"
        
        return advice

