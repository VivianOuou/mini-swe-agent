"""Stage 0: harness checks with no real model. Verifies commit/event/budget invariants from the V0 plan."""

import pytest

from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.test_models import DeterministicModel, make_output
from orchestrator.artifacts import ArtifactState, Evidence, Hypothesis, Patch, TestResult
from orchestrator.events import Event, EventQueue, EventType, derive_event
from orchestrator.scheduler import Budget, Mode, Scheduler
from orchestrator.workers import BudgetedModel


def new_state() -> ArtifactState:
    return ArtifactState(task_id="t", issue="bug", base_commit="abc")


def ready_state() -> ArtifactState:
    state = new_state()
    state.commit(Evidence("e1", "logs/e1.txt", "r0"))
    state.commit(Hypothesis("h1", "off by one", ["e1"], ["a.py"]))
    return state


def result(id: str, patch_id: str, revision: str, scope: str, exit_code: int = 0) -> TestResult:
    return TestResult(id, patch_id, revision, scope, f"pytest -k {scope}", exit_code, f"logs/{id}.txt")


def usage_output(tokens: int) -> dict:
    out = make_output("ok", [{"command": "true"}])
    out["extra"]["response"] = {"usage": {"prompt_tokens": tokens - 10, "completion_tokens": 10}}
    return out


def test_commit_rejects_dangling_evidence_and_stale_results():
    state = new_state()
    with pytest.raises(ValueError, match="unknown evidence"):
        state.commit(Hypothesis("h1", "x", ["missing"], ["a.py"]))
    state = ready_state()
    state.commit(Patch("p1", "h1", "r0", "diff", ["e1"]), repo_revision="r1")
    with pytest.raises(ValueError, match="current revision is r1"):
        state.commit(result("t1", "p1", "r0", "repro"))
    with pytest.raises(ValueError, match="based on r0"):
        state.commit(Patch("p2", "h1", "r0", "diff", ["e1"]))
    assert (state.state_version, len(state.test_results), len(state.patches)) == (3, 0, 1)


def test_event_flow_follows_the_four_event_rules(tmp_path):
    state = new_state()
    assert derive_event(state).type == EventType.NEED_EVIDENCE
    state.commit(Evidence("e1", "logs/e1.txt", "r0"))
    state.commit(Hypothesis("h0", "no target yet", ["e1"]))
    assert derive_event(state).type == EventType.NEED_EVIDENCE
    state.commit(Hypothesis("h1", "off by one", ["e1"], ["a.py"]))
    assert derive_event(state).type == EventType.READY_TO_PATCH
    state.commit(Patch("p1", "h1", "r0", "diff", ["e1"]), repo_revision="r1")
    assert derive_event(state) is None
    state.commit(result("t1", "p1", "r1", "repro"))
    assert derive_event(state) is None  # regression check still outstanding
    state.commit(result("t2", "p1", "r1", "regression", exit_code=1))
    assert derive_event(state).type == EventType.PATCH_FAILED
    state.commit(Hypothesis("h2", "wrong module", ["e1"]))
    assert derive_event(state).type == EventType.NEED_EVIDENCE
    state.save(tmp_path / "state.json")
    assert ArtifactState.load(tmp_path / "state.json") == state


def test_stale_pass_cannot_solve_a_new_patch():
    state = ready_state()
    state.commit(Patch("p1", "h1", "r0", "diff", ["e1"]), repo_revision="r1")
    state.commit(result("t1", "p1", "r1", "repro"), result("t2", "p1", "r1", "regression"))
    assert derive_event(state).type == EventType.SOLVED
    state.commit(Patch("p2", "h1", "r1", "diff2", ["e1"]), repo_revision="r2")
    assert derive_event(state) is None
    state.commit(result("t3", "p2", "r2", "repro"))
    assert derive_event(state) is None
    state.commit(result("t4", "p2", "r2", "regression"))
    assert derive_event(state).type == EventType.SOLVED


def test_apply_failure_is_patch_failed_and_queue_dedups_and_drops_stale(tmp_path):
    state = ready_state()
    state.commit(Patch("p1", "h1", "r0", "bad diff", ["e1"], applied=False))
    event = derive_event(state)
    queue = EventQueue(tmp_path / "events.jsonl")
    assert (event.type, queue.push(event), queue.push(event)) == (EventType.PATCH_FAILED, True, False)
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1
    state.commit(Evidence("e2", "logs/e2.txt", "r0"))
    assert queue.pop(state) is None


def test_queue_pops_by_priority_within_one_version():
    state = ready_state()
    queue = EventQueue()
    for type in (EventType.READY_TO_PATCH, EventType.PATCH_FAILED, EventType.SOLVED):
        queue.push(Event(type, state.state_version, "r0", "x"))
    assert [queue.pop(state).type for _ in range(3)] == [
        EventType.SOLVED,
        EventType.PATCH_FAILED,
        EventType.READY_TO_PATCH,
    ]


def test_scheduler_maps_modes_stops_on_stall_budget_and_solved(tmp_path):
    state, budget = ready_state(), Budget()
    scheduler = Scheduler(budget, max_repeats=2, log_path=tmp_path / "decisions.jsonl")
    need = Event(EventType.NEED_EVIDENCE, 1, "r0", "same")
    assert [scheduler.decide(need, state).mode for _ in range(3)] == [Mode.EXPLORE, Mode.EXPLORE, None]
    assert scheduler.decide(Event(EventType.NEED_EVIDENCE, 2, "r0", "other"), state).mode == Mode.EXPLORE
    assert scheduler.decide(Event(EventType.PATCH_FAILED, 3, "r0", "f"), state).mode == Mode.DIAGNOSE
    assert scheduler.decide(Event(EventType.SOLVED, 4, "r0", "p1"), state).reason == "solved"
    budget.tokens = budget.max_tokens
    assert scheduler.decide(Event(EventType.READY_TO_PATCH, 5, "r0", "h1"), state).reason == "budget_exhausted"
    assert len((tmp_path / "decisions.jsonl").read_text().splitlines()) == 7


def test_budget_is_shared_across_mode_switches_and_never_resets():
    budget = Budget(max_tokens=1_000, max_calls=80, max_activation_calls=2)
    model = BudgetedModel(
        DeterministicModel(outputs=[usage_output(300), usage_output(300), usage_output(300), usage_output(300)]),
        budget,
        reserve_tokens=100,
    )
    model.begin_activation()
    model.query([])
    model.query([])
    with pytest.raises(LimitsExceeded):
        model.query([])  # per-activation cap
    model.begin_activation()  # next mode: calls reset, token budget does not
    model.query([])
    assert (budget.tokens, budget.calls) == (900, 3)


def test_reserve_blocks_a_call_that_could_overshoot():
    budget = Budget(max_tokens=1_000)
    model = BudgetedModel(DeterministicModel(outputs=[usage_output(900), usage_output(50)]), budget, reserve_tokens=200)
    model.query([])
    with pytest.raises(LimitsExceeded):
        model.query([])
    assert (budget.tokens, budget.calls) == (900, 1)


def test_format_errors_and_missing_usage_are_still_charged():
    budget = Budget()
    bad = {
        "role": "assistant",
        "content": "x" * 400,
        "extra": {"response": {"usage": {"prompt_tokens": 70, "completion_tokens": 30}}},
    }
    no_usage = make_output("y" * 400, [{"command": "true"}])
    model = BudgetedModel(
        DeterministicModel(outputs=[{"extra": {"actions": [{"raise": FormatError(bad)}]}}, no_usage]), budget
    )
    with pytest.raises(FormatError):
        model.query([])
    model.query([])
    assert (budget.tokens, budget.calls, budget.estimated_calls) == (200, 2, 1)
