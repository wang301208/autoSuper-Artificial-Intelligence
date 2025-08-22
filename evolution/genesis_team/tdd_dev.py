"""TDD developer agent builds features using tests."""

from __future__ import annotations

import subprocess

from .. import Agent


class TDDDeveloper(Agent):
    """Executes tests to drive development."""

    def perform(self, test_cmd: str = "pytest") -> str:
        try:
            result = subprocess.run(
                test_cmd.split(), capture_output=True, text=True, check=False
            )
            return result.stdout + result.stderr
        except Exception as err:
            return f"Test execution failed: {err}"
