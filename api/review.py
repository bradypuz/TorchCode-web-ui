"""Explainable review priorities, using a heuristic exponential forgetting curve.

This is not a fitted memory model. The recall equation follows half-life
regression, while intervals and signal weights are deliberately simple product
defaults that can be inspected and changed without training data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any


INTERVALS = (2, 4, 7, 14, 30)
RECALL_TARGET = 0.8


def mistake_severity(attempt: dict[str, Any]) -> float:
    if attempt["success"]:
        return 0.25 if attempt.get("hint_used") else 0.0
    total = max(0, int(attempt.get("total", 0)))
    if total == 0:
        return 1.0
    passed = min(total, max(0, int(attempt.get("passed", 0))))
    # A failed submission still signals difficulty if its summary has no
    # individual failing tests (for example a run-level validation failure).
    return max(0.1, 1.0 - passed / total)


def recent_mistakes(attempts: list[dict[str, Any]]) -> float:
    recent = list(reversed(attempts[-5:]))
    if not recent:
        return 0.0
    weights = [0.5 ** index for index in range(len(recent))]
    return sum(weight * mistake_severity(attempt)
               for weight, attempt in zip(weights, recent)) / sum(weights)


def memory_stage(attempts: list[dict[str, Any]]) -> int:
    """Count successful spaced recall evidence, including older real history.

    Re-deriving this also corrects databases whose old counter rewarded many
    successful submissions minutes apart. A success after two or more failures
    starts with the shorter, one-day confidence-building interval.
    """
    stage = failures = 0
    last_success = None
    for attempt in attempts:
        if not attempt["success"]:
            stage = max(0, stage - 1)
            failures += 1
            continue
        at = datetime.fromisoformat(attempt["created_at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if attempt.get("hint_used") or failures >= 2:
            stage = 0
        else:
            spaced = last_success is None or (at - last_success).total_seconds() >= 86400
            stage = max(1, stage + int(spaced))
        last_success = at
        failures = 0
    return stage


def review_metrics(state: dict[str, Any], attempts: list[dict[str, Any]],
                   as_of: datetime, *, unknown_memory_due: bool = False) -> dict[str, Any]:
    manual = bool(state.get("manual_review"))
    mistakes = recent_mistakes(attempts)
    successes = [attempt for attempt in attempts if attempt["success"]]
    retention = half_life = None
    forgetting = float(unknown_memory_due) if state.get("solved") else 0.0
    if successes:
        anchor = datetime.fromisoformat(successes[-1]["created_at"])
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        elapsed_days = max(0.0, (as_of - anchor).total_seconds() / 86400)
        stage = memory_stage(attempts)
        interval = INTERVALS[min(stage - 1, len(INTERVALS) - 1)] if stage else 1
        # The chosen interval reaches 80% retention without mistakes. Recent
        # difficulty reduces memory strength; failures never become an anchor.
        half_life = interval / -math.log2(RECALL_TARGET) * (1.0 - 0.6 * mistakes)
        retention = 2.0 ** (-elapsed_days / half_life)
        forgetting = 1.0 - retention
    score = 100 * (0.40 * manual + 0.35 * mistakes + 0.25 * forgetting)
    return {
        "review_score": round(min(100.0, max(0.0, score)), 2),
        "estimated_retention": round(retention, 4) if retention is not None else None,
        "half_life_days": round(half_life, 4) if half_life is not None else None,
        "review_factors": {"manual": manual, "mistakes": round(mistakes, 4),
                           "forgetting": round(forgetting, 4)},
    }
