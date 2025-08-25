"""Utilities for unified exception logging and handling."""
from __future__ import annotations

import logging
from typing import Optional


def handle_exception(
    exc: Exception,
    logger: Optional[logging.Logger] = None,
    propagate: bool = False,
) -> None:
    """Log *exc* and optionally re-raise it.

    Args:
        exc: The exception instance to handle.
        logger: Optional logger to use; if not provided, a module-level logger is
            used.
        propagate: When ``True`` the exception is re-raised after logging.
    """

    log = logger or logging.getLogger(__name__)
    log.exception("%s", exc)
    if propagate:
        raise exc
