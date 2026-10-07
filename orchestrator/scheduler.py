"""Frozen rule-based scheduler and the task-level budget shared by every mode activation."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from orchestrator.artifacts import ArtifactState
from orchestrator.events import Event, EventType


class Mode(str, Enum):
    EXPLORE = "Explore"
    PATCH = "Patch"
    DIAGNOSE = "Diagnose"


MODE_FOR_EVENT = {
    EventType.NEED_EVIDENCE: Mode.EXPLORE,
    EventType.READY_TO_PATCH: Mode.PATCH,
    EventType.PATCH_FAILED: Mode.DIAGNOSE,
}


@dataclass
class Budget:
    max_tokens: int = 2_000_000
    max_calls: int = 150
    max_activation_calls: int = 20
    reserve_tokens: int = 4096
    """Completion allowance that must still fit under max_tokens before another call is allowed."""
    tokens: int = 0
    calls: int = 0
    estimated_calls: int = 0
    """Calls whose usage was missing from the provider response and was estimated instead."""
    cost_usd: float = 0.0

    def can_call(self, pending_tokens: int, activation_calls: int, activation_cap: int | None = None) -> bool:
        """pending_tokens is the estimated size of the prompt about to be sent."""
        return (
            self.calls < self.max_calls
            and self.tokens + pending_tokens + self.reserve_tokens <= self.max_tokens
            and activation_calls < (activation_cap or self.max_activation_calls)
        )

    def charge(self, tokens: int, estimated: bool = False, cost_usd: float = 0.0) -> None:
        self.tokens += tokens
        self.calls += 1
        self.estimated_calls += estimated
        self.cost_usd += cost_usd

    @property
    def exhausted(self) -> bool:
        """Same predicate as can_call with an empty prompt, so the stop reason matches why calls are refused."""
        return not self.can_call(0, 0)


@dataclass
class Decision:
    event_id: str
    mode: Mode | None
    """None means stop."""
    reason: str
    state_version: int
    budget: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, budget: Budget, max_repeats: int = 2, log_path: Path | None = None):
        self.budget = budget
        self.max_repeats = max_repeats
        self.log_path = log_path
        self.repeats: dict[tuple, int] = {}

    def decide(self, event: Event, state: ArtifactState) -> Decision:
        """Priority is already resolved by the queue. Rules: solved > budget > no-progress cap > event->mode."""
        key = (event.type, event.fingerprint)
        if event.type == EventType.SOLVED:
            mode, reason = None, "solved"
        elif self.budget.exhausted:
            mode, reason = None, "budget_exhausted"
        elif self.repeats.get(key, 0) >= self.max_repeats:
            mode, reason = None, "stalled"
        else:
            self.repeats[key] = self.repeats.get(key, 0) + 1
            mode, reason = MODE_FOR_EVENT[event.type], event.type.value
        decision = Decision(event.id, mode, reason, state.state_version, asdict(self.budget))
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(asdict(decision)) + "\n")
        return decision
