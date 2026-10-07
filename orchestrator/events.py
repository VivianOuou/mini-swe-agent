"""The four V0 domain events, derived only from committed ArtifactState, plus a deduplicating queue/log."""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from orchestrator.artifacts import ArtifactState

REQUIRED_CHECKS = ("repro", "regression")


class EventType(str, Enum):
    SOLVED = "Solved"
    PATCH_FAILED = "PatchFailed"
    NEED_EVIDENCE = "NeedEvidence"
    READY_TO_PATCH = "ReadyToPatch"


PRIORITY = {EventType.SOLVED: 0, EventType.PATCH_FAILED: 1, EventType.NEED_EVIDENCE: 2, EventType.READY_TO_PATCH: 3}


@dataclass
class Event:
    type: EventType
    state_version: int
    revision: str
    fingerprint: str

    @property
    def id(self) -> str:
        return f"{self.type.value}-{self.state_version}"

    @property
    def key(self) -> tuple:
        return (self.type, self.state_version, self.fingerprint)


def derive_event(state: ArtifactState, required: tuple[str, ...] = REQUIRED_CHECKS) -> Event | None:
    """Return the event implied by the current state, or None while a candidate patch awaits its checks.

    The fingerprint identifies a failure, not a hypothesis: the same failure with the same amount of evidence repeats the
    fingerprint however often the hypothesis is rewritten, which is what the scheduler's no-progress stop counts."""
    patch = state.get(state.patches, state.active_patch_id)
    hyp = state.get(state.hypotheses, state.active_hypothesis_id)
    n_evidence = len(state.evidence)

    def event(type: EventType, *parts) -> Event:
        return Event(type, state.state_version, state.repo_revision, ":".join(map(str, (*parts, n_evidence))))

    if patch and hyp and patch.hypothesis_id == hyp.id:
        latest = {
            r.scope: r
            for r in state.test_results
            if r.patch_id == patch.id and r.tested_revision == state.repo_revision
        }
        if not patch.applied:
            return event(EventType.PATCH_FAILED, "apply")
        if failed := next((r for r in latest.values() if r.exit_code != 0), None):
            return event(EventType.PATCH_FAILED, failed.scope, failed.exit_code)
        if all(scope in latest for scope in required):
            return event(EventType.SOLVED, patch.id)
        return None
    if hyp and hyp.evidence_ids and hyp.target_files:
        return event(EventType.READY_TO_PATCH, hyp.id)
    return event(EventType.NEED_EVIDENCE, "need")


class EventQueue:
    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path
        self.pending: list[Event] = []
        self.seen: set[tuple] = set()

    def push(self, event: Event) -> bool:
        """Returns False for an event with an already-seen (type, version, fingerprint)."""
        if event.key in self.seen:
            return False
        self.seen.add(event.key)
        self.pending.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(asdict(event) | {"id": event.id}) + "\n")
        return True

    def pop(self, state: ArtifactState) -> Event | None:
        """Highest-priority event that is still valid for the current state version; stale ones are dropped."""
        self.pending = [e for e in self.pending if e.state_version == state.state_version]
        if not self.pending:
            return None
        best = min(self.pending, key=lambda e: PRIORITY[e.type])
        self.pending.remove(best)
        return best
