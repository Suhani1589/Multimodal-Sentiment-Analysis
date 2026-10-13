from __future__ import annotations

def build_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
