from __future__ import annotations
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ProgressHeartbeat:
    """Dependency-free atomic progress publisher for data-processing jobs."""

    def __init__(self, path: Path, *, interval_seconds: float=30.0) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._started = time.monotonic()
        self._stage = 'initializing'
        self._detail = ''
        self._completed: int | None = None
        self._total: int | None = None
        self._extra: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self, status: str='running') -> None:
        with self._lock:
            completed = self._completed
            total = self._total
            payload = {'schema_version': 1, 'producer_module': 'hpsafe_features', 'status': status, 'heartbeat_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': round(time.monotonic() - self._started, 3), 'stage': self._stage, 'detail': self._detail, 'completed': completed, 'total': total, 'percent': None if completed is None or total in {None, 0} else round(100.0 * completed / total, 2), **self._extra}
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, suffix='.tmp', delete=False) as handle:
                temporary = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write('\n')
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def update(self, stage: str, *, detail: str='', completed: int | None=None, total: int | None=None, **extra: Any) -> None:
        with self._lock:
            self._stage = str(stage)
            self._detail = str(detail)
            self._completed = completed
            self._total = total
            self._extra.update(extra)
        self._write()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write()

    def __enter__(self) -> ProgressHeartbeat:
        self._write()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.interval_seconds))
        with self._lock:
            self._stage = 'complete' if exc is None else 'failed'
            self._detail = '' if exc is None else repr(exc)
        self._write(status='complete' if exc is None else 'failed')
