from .llama import LLAMA_MODEL

# Keywords that identify a Llama-family model (HF repo name or local path)
_LLAMA_KEYWORDS = ("llama", "Llama", "LLAMA")

def load_lm_module(lm_type, device, attn_implementation, dtype):
    """Load the appropriate LM module based on ``lm_type``.

    ``lm_type`` can be either a HuggingFace model ID (e.g.
    ``"meta-llama/Llama-3.2-1B-Instruct"``) or a local directory path
    (e.g. ``"/home/user/Llama-3.2-1B-Instruct"``).  The check is
    keyword-based so both forms are accepted.
    """
    if any(kw in lm_type for kw in _LLAMA_KEYWORDS):
        return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    else:
        raise ValueError(
            f"Unsupported LM type: {lm_type!r}. "
            "Model name or path must contain one of: " + ", ".join(_LLAMA_KEYWORDS)
        )
