""" CLIP tokenizer (trimmed for Vir2_CLIP inference).

Only the HuggingFace tokenizer path (``HFTokenizer``) is retained — the OpenAI
BPE ``SimpleTokenizer`` and SigLIP/nltk masking helpers have been removed since
Vir2_CLIP uses a BioMedBERT HF tokenizer exclusively.

Derived from https://github.com/openai/CLIP (MIT License, Copyright (c) 2021 OpenAI).
"""
import html
import os
import string
from typing import List, Optional, Union
import warnings

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize_text(text, *, keep_punctuation_exact_string=None):
    """Returns canonicalized `text` (lowercase and punctuation removed)."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_canonicalize(x):
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def get_hf_tokenizer(local_model_path: str, context_length: int = DEFAULT_CONTEXT_LENGTH) -> 'HFTokenizer':
    """Get HuggingFace tokenizer"""
    return HFTokenizer(local_model_path, context_length=context_length)


def get_hf_bert_tokenizer(hf_bert_model_path: str):
    """Get HuggingFace Bert tokenizer"""
    return AutoTokenizer.from_pretrained(hf_bert_model_path)


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids

        if self.strip_sep_token and (self.tokenizer.sep_token_id is not None):
            input_ids = torch.where(
                input_ids == self.tokenizer.sep_token_id,
                torch.zeros_like(input_ids),
                input_ids,
            )

        return input_ids

    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')
