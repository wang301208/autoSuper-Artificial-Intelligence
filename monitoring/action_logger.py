import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ActionLogger:
    """Append structured action logs in JSON Lines format."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append *record* to the log with timestamp and unique id."""
        record.setdefault("id", str(uuid.uuid4()))
        record.setdefault("timestamp", datetime.utcnow().isoformat())
        with self.log_path.open("a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")
