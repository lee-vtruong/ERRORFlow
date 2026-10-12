"""Experiment manifests and contamination-safe run layout."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_json, config_hash


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class RunManifest:
    run_name: str
    config: dict[str, Any]
    dataset: str
    split: str
    seed: int
    status: str = "created"
    schema_version: int = 2
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = field(default_factory=lambda: git_value("rev-parse", "HEAD"))
    git_dirty: bool | None = field(default_factory=lambda: bool(git_value("status", "--porcelain")))
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def config_id(self) -> str:
        return config_hash(self.config)

    @property
    def run_id(self) -> str:
        return f"{self.run_name}-{self.dataset}-{self.split}-s{self.seed}-{self.config_id}"

    def save(self, root: str | Path) -> Path:
        directory = Path(root) / self.run_id
        path = directory / "manifest.json"
        payload = asdict(self)
        payload["config_id"] = self.config_id
        payload["run_id"] = self.run_id
        atomic_write_json(path, payload)
        return path


def validate_memory_boundary(current_split: str, allowed_memory_splits: set[str]) -> None:
    forbidden = {"dev", "development", "validation", "test"}
    if current_split.lower() in forbidden and forbidden & {item.lower() for item in allowed_memory_splits}:
        raise ValueError("Evaluation memory may not include dev/validation/test splits")
