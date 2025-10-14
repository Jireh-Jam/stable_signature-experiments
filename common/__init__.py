"""Common utilities shared across advanced_attacks and detector.

This package holds cross-cutting helpers (logging, IO, config, transforms registry)
that can be safely imported from both folders.
"""

from .logging_utils import get_logger  # re-export for convenience
