from __future__ import annotations

import hashlib
from typing import Any, Dict

from autogpts.autogpt.autogpt.core.errors import AutoGPTError


class SkillSecurityError(AutoGPTError):
    """Raised when a skill fails security verification."""

    def __init__(self, skill: str, cause: str) -> None:
        super().__init__(f"Skill {skill} blocked: {cause}")
        self.skill = skill
        self.cause = cause


SAFE_BUILTINS: Dict[str, Any] = {
    "__import__": __import__,
    "len": len,
    "range": range,
    "print": print,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
}


def _verify_skill(name: str, code: str, metadata: Dict[str, Any]) -> None:
    """Ensure skill source passes signature verification."""

    signature = metadata.get("signature")
    if not signature:
        raise SkillSecurityError(name, "missing signature")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if signature != digest:
        raise SkillSecurityError(name, "invalid signature")
