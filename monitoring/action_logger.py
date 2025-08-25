import json
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ActionLogger:
    """Append structured action logs in JSON Lines format.

    A dedicated writer thread consumes log records from a queue to avoid
    contention between producer threads. A ``threading.Lock`` guards writes to
    the underlying file to prevent interleaving when multiple batches are
    flushed concurrently.
    """

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._queue: queue.Queue[Dict[str, Any] | None] = queue.Queue()
        self._worker = threading.Thread(target=self._writer, daemon=True)
        self._worker.start()

    def _writer(self) -> None:
        """Background worker that writes queued records in batches."""
        buffer: list[Dict[str, Any]] = []
        while True:
            item = self._queue.get()
            if item is None:
                break
            buffer.append(item)
            # Drain the queue to batch write multiple records at once.
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        # Sentinel encountered; push back for shutdown and stop
                        self._queue.put(None)
                        break
                    buffer.append(item)
            except queue.Empty:
                pass

            with self._lock, self.log_path.open("a", encoding="utf-8") as f:
                for rec in buffer:
                    json.dump(rec, f)
                    f.write("\n")
            buffer.clear()

    def log(self, record: Dict[str, Any]) -> None:
        """Queue *record* to be appended to the log with timestamp and unique id."""
        record.setdefault("id", str(uuid.uuid4()))
        record.setdefault("timestamp", datetime.utcnow().isoformat())
        self._queue.put(record)

    def close(self) -> None:
        """Shut down the writer thread, flushing any pending logs."""
        self._queue.put(None)
        self._worker.join()
