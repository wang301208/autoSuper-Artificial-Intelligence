from __future__ import annotations

import logging
from pathlib import Path

import structlog
import yaml

_configured = False


def get_logger(name: str) -> structlog.BoundLogger:
    global _configured
    if not _configured:
        config_path = Path(__file__).resolve().parents[2] / "config" / "logging.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        level_name = cfg.get("level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(level=level, format=cfg.get("format"))
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.KeyValueRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        _configured = True
    return structlog.get_logger(name)
