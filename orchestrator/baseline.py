"""B0: the native single mini agent, run under exactly the same budget, wrapper and patch extraction as EDAC."""

import json
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from orchestrator.checks import final_patch, init_workspace, quarantine_strays
from orchestrator.scheduler import Budget
from orchestrator.workers import BudgetedModel


def run_b0_task(
    *, task_id: str, issue: str, env, model, budget: Budget, out_dir: Path, agent_config: dict, max_seconds: int = 1800
) -> dict:
    """One continuous native agent loop. Its own submission text is ignored: the patch is taken from the working tree
    with the shared rule, whatever the exit status, so running out of budget does not forfeit edits already made."""
    started = time.monotonic()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    exit_status, errors, patch = "", [], ""
    try:
        base = init_workspace(env)
        bmodel = BudgetedModel(model, budget)
        bmodel.begin_activation()
        agent = DefaultAgent(
            bmodel,
            env,
            **{
                **agent_config,
                "step_limit": 0,
                "cost_limit": 0,
                "wall_time_limit_seconds": max_seconds,
                "output_path": out_dir / "trajectories" / "00_B0.traj.json",
            },
        )
        exit_status = agent.run(issue).get("exit_status", "")
        if moved := quarantine_strays(env, scratch_only=True):
            errors.append(f"moved stray files into .edac/stray/: {moved}")
        patch = final_patch(env, base)
    except Exception:
        # an infrastructure failure must not lose the run's metrics or take the whole batch down with it
        exit_status = "infra_failed"
        errors.append(f"infra failure:\n{traceback.format_exc()}")
    (out_dir / "patch.diff").write_text(patch)
    metrics = {
        "task_id": task_id,
        "run_status": "infra_failed" if exit_status == "infra_failed" else exit_status or "unknown",
        "exit_status": exit_status,
        "patch_chars": len(patch),
        "errors": errors,
        "wall_seconds": round(time.monotonic() - started, 1),
        **{k: v for k, v in asdict(budget).items() if k in ("tokens", "calls", "estimated_calls", "cost_usd")},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
