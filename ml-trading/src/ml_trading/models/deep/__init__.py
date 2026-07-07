"""Deep temporal models (optional `deep` extra: torch).

Import guard: this package imports cleanly without torch; constructing a model
raises a helpful error instead.
"""

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = ["TORCH_AVAILABLE"]
