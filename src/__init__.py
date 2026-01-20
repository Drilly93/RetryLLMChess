"""Chess Challenge evaluation module."""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ChessEvaluator":
        from .evaluate import ChessEvaluator
        return ChessEvaluator
    if name == "load_model_and_tokenizer":
        from .evaluate import load_model_and_tokenizer
        return load_model_and_tokenizer
    if name == "count_parameters":
        from .evaluate import count_parameters
        return count_parameters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ChessEvaluator",
    "load_model_and_tokenizer",
    "count_parameters",
]
