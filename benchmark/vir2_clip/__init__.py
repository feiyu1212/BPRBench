"""Minimal vendored Vir2_CLIP package for BPRBench.

Exposes only the two entry points used by the benchmark:
    - get_vir2_clip(checkpoint, ...) -> (model, preprocess)
    - get_vir2_clip_tokenizer()      -> tokenizer

Source derived from the hipa_clip Vir2_CLIP implementation, trimmed to the
modules required for inference. The full open_clip-style API is intentionally
not re-exported.
"""
from .factory import get_vir2_clip, get_vir2_clip_tokenizer

__all__ = ["get_vir2_clip", "get_vir2_clip_tokenizer"]
