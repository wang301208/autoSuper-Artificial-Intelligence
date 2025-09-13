from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.interpretability import InterpretabilityAnalyzer


def test_interpretability_tools(tmp_path: Path) -> None:
    analyzer = InterpretabilityAnalyzer()
    curve_path = tmp_path / "curve.png"
    analyzer.generate_learning_curve([0.1, 0.2, 0.3], str(curve_path))
    assert curve_path.exists()

    analyzer.log_failure_case("input", "output", "expected")
    report_path = tmp_path / "failures.csv"
    analyzer.export_failure_cases(str(report_path))
    assert report_path.exists()
