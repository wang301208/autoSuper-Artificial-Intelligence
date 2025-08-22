import asyncio
import logging
import re
import time
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class EvaluateMetrics(Ability):
    """Evaluate code complexity and runtime for a Python file."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.EvaluateMetrics",
        ),
        workspace_required=True,
    )

    def __init__(self, logger: logging.Logger, workspace: Workspace) -> None:
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = (
        "Evaluate code metrics like cyclomatic complexity and execution time for a Python file."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relative path to the Python file to analyse.",
            required=True,
        )
    }

    async def __call__(self, file_path: str) -> AbilityResult:
        file_abs = self._workspace.get_path(file_path)
        metrics_parts: list[str] = []
        success = True
        try:
            source = file_abs.read_text()
            try:
                from radon.complexity import cc_visit

                blocks = cc_visit(source)
                if blocks:
                    avg_complexity = sum(b.complexity for b in blocks) / len(blocks)
                else:
                    avg_complexity = 0.0
                metrics_parts.append(f"complexity={avg_complexity:.2f}")
            except Exception as e:  # pragma: no cover - best effort
                success = False
                metrics_parts.append(f"complexity_error={e}")
        except Exception as e:  # pragma: no cover - best effort
            success = False
            metrics_parts.append(f"file_error={e}")
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"file_path": file_path},
                success=False,
                message=", ".join(metrics_parts),
            )

        try:
            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(file_abs),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            runtime = time.perf_counter() - start
            metrics_parts.append(f"runtime={runtime:.4f}")
            if proc.returncode != 0:
                success = False
        except Exception as e:  # pragma: no cover - best effort
            success = False
            metrics_parts.append(f"runtime_error={e}")

        # Run pytest with coverage and parse coverage percentage
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest",
                "--cov",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._workspace.root),
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode()
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            if match:
                metrics_parts.append(f"coverage={match.group(1)}%")
            else:
                metrics_parts.append("coverage_error=unparsed")
                success = False
            if proc.returncode != 0:
                success = False
        except FileNotFoundError:
            metrics_parts.append("coverage_error=pytest not installed")
            success = False
        except Exception as e:  # pragma: no cover - best effort
            metrics_parts.append(f"coverage_error={e}")
            success = False

        # Run static analysis (ruff) and count errors
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                str(file_abs),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode().strip()
            if output:
                error_count = len(output.splitlines())
                metrics_parts.append(f"style_errors={error_count}")
                success = False
            else:
                metrics_parts.append("style_errors=0")
            if proc.returncode not in (0, 1):
                success = False
        except FileNotFoundError:
            metrics_parts.append("style_error=ruff not installed")
            success = False
        except Exception as e:  # pragma: no cover - best effort
            metrics_parts.append(f"style_error={e}")
            success = False

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"file_path": file_path},
            success=success,
            message=", ".join(metrics_parts),
        )
