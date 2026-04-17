"""
database.py — Lightweight in-memory storage layer.

Uses a thread-safe dictionary to persist workflow inputs and results
for the lifetime of the application process.  Swap this module for a
real database adapter (SQLAlchemy / asyncpg) when moving to production.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[UUID, dict[str, Any]] = {}

    def create(self, input_id: UUID, text: str) -> dict[str, Any]:
        record = {
            "input_id": str(input_id),
            "text": text,
            "status": "pending",
            "result": None,
            "steps": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        with self._lock:
            self._records[input_id] = record
        return record

    def get(self, input_id: UUID) -> dict[str, Any] | None:
        with self._lock:
            return self._records.get(input_id)

    def update(
        self,
        input_id: UUID,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(input_id)
            if record is None:
                return None
            if status is not None:
                record["status"] = status
            if result is not None:
                record["result"] = result
            if steps is not None:
                record["steps"] = steps
            if status == "completed":
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
            return record

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records.values())


store = InMemoryStore()
