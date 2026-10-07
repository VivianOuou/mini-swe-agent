"""Single-task EDAC V0 control loop: state -> event -> frozen scheduler -> bounded worker -> real checks -> state."""

import json
import shutil
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from orchestrator.artifacts import ArtifactState, Evidence, Patch
from orchestrator.checks import (
    FORBIDDEN_PATH,
    INVALID_REPRO_CODES,
    SETUP_ERROR,
    apply_diff,
    changed_files,
    init_workspace,
    quarantine_strays,
    reset_to_base,
    restore_head,
    run_check,
    run_visible_checks,
    snapshot_edac,
    source_diff,
)
from orchestrator.events import EventQueue, derive_event
from orchestrator.prompts import state_view
from orchestrator.scheduler import Budget, Mode, Scheduler
from orchestrator.workers import BudgetedModel, HandoffError, bind_handoff, parse_handoff, run_activation

STOP_TO_STATUS = {
    "solved": "local_success",
    "budget_exhausted": "budget_exhausted",
    "time_exhausted": "budget_exhausted",
    "check_limit": "budget_exhausted",
    "stalled": "stalled",
    "infra_failed": "infra_failed",
}
REPAIR_CAP = 3


def run_task(
    *,
    task_id: str,
    issue: str,
    env,
    model,
    budget: Budget,
    out_dir: Path,
    python: str = "python",
    max_check_batches: int = 10,
    check_timeout: int = 180,
    max_seconds: int = 1800,
) -> dict:
    """Runs one task serially (one worker, one candidate patch). Returns the metrics dict that is also saved.

    A rerun starts from an empty out_dir. Whatever happens, state, patch and metrics are written before returning.
    """
    started = time.monotonic()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    state = ArtifactState(task_id, issue, "")
    queue = EventQueue(out_dir / "events.jsonl")
    scheduler = Scheduler(budget, log_path=out_dir / "decisions.jsonl")
    bmodel = BudgetedModel(model, budget)
    counts: Counter = Counter()
    stats: Counter = Counter()
    frozen: dict[str, str] = {}
    revision = 0
    stop_reason = "stalled"

    def new_revision() -> str:
        nonlocal revision
        revision += 1
        return f"r{revision}"

    def activate(mode: Mode, repair=None, suffix: str = "") -> tuple[dict, str]:
        path = out_dir / "trajectories" / f"{sum(counts.values()):02d}_{mode.value}{suffix}.traj.json"
        act = run_activation(
            mode=mode,
            model=bmodel,
            env=env,
            task=issue,
            view=state_view(state, mode, source_diff(env, state.base_commit)),
            has_repro="repro" in state.checks,
            traj_path=path,
            cap=REPAIR_CAP if repair else None,
            repair=repair,
        )
        return act, str(path.relative_to(out_dir))

    def contain(mode: Mode, before: str) -> None:
        """Undo whatever a worker did outside its mode's remit, anchored to the base commit rather than HEAD."""
        base = state.base_commit
        if restore_head(env, base):
            state.errors.append(f"{mode.value}: worker moved HEAD (git commit); HEAD restored, working tree kept")
        if moved := quarantine_strays(env, scratch_only=mode == Mode.PATCH):
            state.errors.append(f"{mode.value}: moved stray files into .edac/stray/: {moved}")
        if mode != Mode.PATCH and source_diff(env, base) != before:
            state.errors.append(f"{mode.value} modified source files; changes reverted")
            reset_to_base(env, base)
            if before and not apply_diff(env, before):
                raise RuntimeError("failed to restore the candidate patch after a read-only violation")

    def validate_repro(data: dict, evidence_number: int) -> Evidence:
        cmd = str(data.get("repro_cmd") or "").strip()
        if not cmd:
            raise HandoffError('the first Explore handoff must include "repro_cmd"')
        log = out_dir / "checks" / "repro_base.log"
        code = run_check(env, cmd, log, check_timeout)
        stats["batches"] += 1
        if code in INVALID_REPRO_CODES or SETUP_ERROR.search(log.read_text()):
            raise HandoffError(
                f"repro_cmd does not reproduce the issue on the unmodified repository (exit code {code}): it must fail "
                "because of the issue, not because it passed, timed out or could not run; keep scripts under .edac/"
            )
        frozen.update(snapshot_edac(env))
        return Evidence(
            f"e{evidence_number}",
            str(log.relative_to(out_dir)),
            state.repo_revision,
            f"repro fails at base (exit {code})",
        )

    def explore_or_diagnose(mode: Mode) -> None:
        before = source_diff(env, state.base_commit)
        act, ref = activate(mode)
        contain(mode, before)
        commands, records, data, raw, last, errors = act["commands"], [], {}, act["submission"], act, []
        for attempt in range(2):
            try:
                if not raw:
                    raise HandoffError(f"no handoff submitted (exit status {last['exit_status'] or 'unknown'})")
                data = parse_handoff(raw)
                records = bind_handoff(data, state, commands, env, ref)
                if "repro" not in state.checks:
                    records.append(
                        validate_repro(data, len(state.evidence) + sum(isinstance(r, Evidence) for r in records) + 1)
                    )
                errors = []
                break
            except HandoffError as e:
                errors.append(str(e))
                records = []
            if attempt == 1 or not raw:
                break
            stats["repairs"] += 1
            before = source_diff(env, state.base_commit)
            last, ref2 = activate(mode, repair=(errors[-1], raw, commands), suffix="_repair")
            contain(mode, before)
            commands, raw, ref = commands + last["commands"], last["submission"], f"{ref}+{ref2}"
        if error := " -> after repair: ".join(errors):
            state.errors.append(f"{mode.value}: {error}")
        elif records:
            if "repro" not in state.checks:
                state.checks["repro"] = data["repro_cmd"].strip()
            state.unresolved_questions = [str(q) for q in data.get("unresolved") or []]
        state.commit(*records)

    def patch_mode() -> bool:
        """Returns False when the check-batch limit stops the run."""
        base = state.base_commit
        if state.patches or source_diff(env, base):
            reset_to_base(env, base)
            state.commit(repo_revision=new_revision())
        activate(Mode.PATCH)
        contain(Mode.PATCH, "")
        diff = source_diff(env, base)
        touched = [f for f in changed_files(env, base) if FORBIDDEN_PATH.search(f)]
        if touched or not diff.strip():
            state.errors.append(f"patch discarded: {'touched ' + str(touched) if touched else 'empty diff'}")
            reset_to_base(env, base)
            state.commit()
            return True
        reset_to_base(env, base)
        hyp = state.get(state.hypotheses, state.active_hypothesis_id)
        pid = f"p{len(state.patches) + 1}"
        patch = Patch(pid, hyp.id, state.repo_revision, diff, list(hyp.evidence_ids), applied=apply_diff(env, diff))
        if not patch.applied:
            state.commit(patch)
            return True
        state.commit(patch, repo_revision=new_revision())
        if stats["batches"] >= max_check_batches:
            return False
        results, used, tampered, vacuous = run_visible_checks(
            env, state, pid, out_dir / "checks", base=base, frozen=frozen, python=python, timeout=check_timeout
        )
        stats["batches"] += used
        stats["vacuous_regression_gates"] += vacuous
        stats["tampered_check_files"] += len(tampered)
        if tampered:
            state.errors.append(f"frozen check files were modified and have been restored before checking: {tampered}")
        state.commit(*results)
        return True

    try:
        state.base_commit = init_workspace(env)
        queue.push(derive_event(state))
        while True:
            event = queue.pop(state)
            if event is None:
                break
            decision = scheduler.decide(event, state)
            if decision.mode is None:
                stop_reason = decision.reason
                break
            if time.monotonic() - started > max_seconds:
                stop_reason = "time_exhausted"
                break
            counts[decision.mode.value] += 1
            if decision.mode == Mode.PATCH:
                if not patch_mode():
                    stop_reason = "check_limit"
                    break
            else:
                explore_or_diagnose(decision.mode)
            if (next_event := derive_event(state)) is not None:
                queue.push(next_event)
    except Exception:
        # an infrastructure failure must not lose the run's state or take the whole batch down with it
        stop_reason = "infra_failed"
        state.errors.append(f"infra failure:\n{traceback.format_exc()}")

    state.run_status = STOP_TO_STATUS[stop_reason]
    state.budget_used = asdict(budget)
    state.save(out_dir / "state.json")
    (out_dir / "frozen_checks.json").write_text(json.dumps(frozen, indent=2))
    final = state.get(state.patches, state.active_patch_id)
    (out_dir / "patch.diff").write_text(final.diff if final else "")
    metrics = {
        "task_id": task_id,
        "run_status": state.run_status,
        "stop_reason": stop_reason,
        "local_solved": state.run_status == "local_success",
        "activations": dict(counts),
        "repair_activations": stats["repairs"],
        "check_batches": stats["batches"],
        "vacuous_regression_gates": stats["vacuous_regression_gates"],
        "tampered_check_files": stats["tampered_check_files"],
        "patches": len(state.patches),
        "errors": state.errors,
        "wall_seconds": round(time.monotonic() - started, 1),
        **{k: v for k, v in asdict(budget).items() if k in ("tokens", "calls", "estimated_calls", "cost_usd")},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
