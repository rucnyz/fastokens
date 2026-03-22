from fastokens._native import Tokenizer

__all__ = ["Tokenizer", "patch_transformers", "unpatch_transformers"]


_patched = False
_originals: dict = {}


def patch_transformers() -> None:
    """
    Monkey-patch ``tokenizers.Tokenizer`` so that the
    ``transformers`` library uses fastokens for encoding.

    Call this before any ``AutoTokenizer.from_pretrained``
    invocation::

        import fastokens
        fastokens.patch_transformers()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B"
        )
    """
    global _patched
    if _patched:
        print("[fastokens] patch_transformers: already patched.")
        return

    from fastokens._compat import _TokenizerShim

    import transformers.tokenization_utils_fast as _tuf

    _originals["TokenizerFast"] = _tuf.TokenizerFast
    _tuf.TokenizerFast = _TokenizerShim
    _patched = True

    from importlib.metadata import version
    # Assuming transformers is installed. 
    # If not, this will raise an error, which is fine since patching won't work without it.
    transformers_version = version("transformers")
    print(f"[fastokens] patch_transformers: successfully patched transformers v{transformers_version}")


def unpatch_transformers() -> None:
    """
    Reverse the monkey-patching applied by :func:`patch_transformers`,
    restoring the ``transformers`` library to its original state.
    """
    global _patched
    if not _patched:
        return

    import transformers.tokenization_utils_fast as _tuf
    from transformers import PreTrainedTokenizerFast

    _tuf.TokenizerFast = _originals["TokenizerFast"]
    PreTrainedTokenizerFast.__call__ = _originals["__call__"]
    PreTrainedTokenizerFast.encode = _originals["encode"]
    PreTrainedTokenizerFast.encode_plus = _originals["encode_plus"]
    PreTrainedTokenizerFast.batch_encode_plus = _originals["batch_encode_plus"]

    _originals.clear()
    _patched = False
