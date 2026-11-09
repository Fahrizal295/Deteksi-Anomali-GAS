"""Small file-based control channel between Streamlit and acquisition service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AcquisitionLock:
    """Cross-platform advisory lock preventing two acquisition processes."""

    def __init__(self, path: str | Path = "data/acquisition_service.lock") -> None:
        self.path = Path(path)
        self._handle = None
        self._windows = os.name == "nt"

    def __enter__(self) -> "AcquisitionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            if self._windows:
                import msvcrt

                self._handle.seek(0)
                self._handle.write("0")
                self._handle.flush()
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"Another acquisition service already holds lock {self.path}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is None:
            return
        try:
            if self._windows:
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def send_command(path: str | Path, command: str, **values: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps({"command": command, **values}), encoding="utf-8")
    os.replace(temporary, target)


def read_command(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        target.unlink(missing_ok=True)
        return payload
    except (OSError, json.JSONDecodeError):
        return None
