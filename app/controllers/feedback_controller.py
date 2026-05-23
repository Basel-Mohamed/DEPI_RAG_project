from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas.feedback import FeedbackRequest

logger = logging.getLogger(__name__)


class FeedbackController:
    feedback_root = Path.cwd() / "feedback"
    feedback_path = feedback_root / "feedback.json"
    feedback_lock = threading.RLock()

    def submit(self, request: FeedbackRequest) -> dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        record = {
            "feedback_id": feedback_id,
            "session_id": request.session_id.strip(),
            "question": request.question.strip(),
            "answer": request.answer.strip(),
            "rating": request.rating,
            "timestamp": request.timestamp.isoformat(),
            "created_at": self._utc_now(),
        }
        self._validate_record(record)

        logger.info(
            "feedback persist started feedback_id=%s session_id=%s rating=%s",
            feedback_id,
            record["session_id"],
            record["rating"],
        )
        with self.feedback_lock:
            feedback = self._read_feedback()
            feedback.append(record)
            self._write_feedback(feedback)
        logger.info("feedback persist completed feedback_id=%s", feedback_id)

        return {
            "feedback_id": feedback_id,
            "session_id": record["session_id"],
            "stored": True,
        }

    def satisfaction_score(self) -> float | None:
        with self.feedback_lock:
            feedback = self._read_feedback()
        if not feedback:
            return None
        positive = sum(1 for item in feedback if item.get("rating") == 1)
        return positive / len(feedback)

    def list_feedback(
        self,
        *,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        with self.feedback_lock:
            feedback = self._read_feedback()
        if session_id:
            feedback = [
                item for item in feedback
                if item.get("session_id") == session_id
            ]
        feedback = sorted(
            feedback,
            key=lambda item: (
                item.get("timestamp", ""),
                item.get("created_at", ""),
            ),
            reverse=True,
        )
        if limit is not None:
            feedback = feedback[:limit]
        return {"feedback": feedback, "total": len(feedback)}

    def satisfaction_summary(self) -> dict[str, Any]:
        with self.feedback_lock:
            feedback = self._read_feedback()
        total = len(feedback)
        positive = sum(1 for item in feedback if item.get("rating") == 1)
        negative = sum(1 for item in feedback if item.get("rating") == -1)
        score = positive / total if total else None
        percent = round(score * 100, 2) if score is not None else None
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_score": score,
            "satisfaction_percent": percent,
        }

    def _read_feedback(self) -> list[dict[str, Any]]:
        if not self.feedback_path.exists():
            return []
        with self.feedback_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Feedback store is corrupted.")
        return data

    def _write_feedback(self, feedback: list[dict[str, Any]]) -> None:
        self.feedback_root.mkdir(parents=True, exist_ok=True)
        temp_path = self.feedback_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(feedback, file, indent=2, sort_keys=True)
        os.replace(temp_path, self.feedback_path)

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> None:
        for field in ("session_id", "question", "answer"):
            if not record[field]:
                raise ValueError(f"{field} is required.")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
