"""Apply parameter suggestions and monitor for regression."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml


def apply_parameters(config_path: Path, params: dict) -> dict:
    config: dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
    config.update(params)
    config_path.write_text(yaml.safe_dump(config))
    return config


def run_tests(command: str) -> tuple[int, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    duration = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    return proc.returncode, duration, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy parameters and run regression tests"
    )
    parser.add_argument(
        "--analytics", default="analytics_output.json", help="Analytics JSON file"
    )
    parser.add_argument(
        "--config", default="prompt_settings.yaml", help="Config file to update"
    )
    parser.add_argument(
        "--test", default="pytest -q", help="Command used for regression tests"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.analytics).read_text())
    params = data.get("suggested_parameters", {})
    apply_parameters(Path(args.config), params)

    code, duration, output = run_tests(args.test)
    report = {"returncode": code, "duration": duration, "output": output}
    Path("deployment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
