"""Small file-based control channel between Streamlit and acquisition service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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
