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
        self.use_auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.use_auth_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token=self.use_auth_token,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Generate advice for the issue."""
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

