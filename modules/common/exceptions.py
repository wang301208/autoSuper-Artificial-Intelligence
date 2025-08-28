"""Project-wide exception utilities."""
from __future__ import annotations

import logging
from typing import Optional


class AutoGPTException(Exception):
    """Base class for AutoGPT project specific exceptions."""


def log_and_format_exception(
    exc: Exception, logger: Optional[logging.Logger] = None
) -> dict[str, str]:
    """Log *exc* and return a standardized error representation."""
    log = logger or logging.getLogger(__name__)
    log.exception("%s: %s", type(exc).__name__, exc)
    return {"error_type": type(exc).__name__, "message": str(exc)}
