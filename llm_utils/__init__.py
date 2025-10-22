"""Utility classes imported from the user's POP project.

The code in this package is copied directly from the original
repository so that it can be imported without requiring the user to
install additional dependencies.  Only a small subset of the POP
project is included here – namely the :class:`PromptFunction` and
several LLM client classes.  See the original POP project for full
documentation and licensing information.
"""

from .POP import PromptFunction  # noqa: F401
from .LLMClient import LLMClient, OpenAIClient, GeminiClient, DeepseekClient, LocalPyTorchClient, DoubaoClient  # noqa: F401

__all__ = [
    "PromptFunction",
    "LLMClient",
    "OpenAIClient",
    "GeminiClient",
    "DeepseekClient",
    "LocalPyTorchClient",
    "DoubaoClient",
]