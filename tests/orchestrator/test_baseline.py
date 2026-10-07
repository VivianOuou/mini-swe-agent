"""B0 runs under the same budget wrapper and patch-extraction rule as EDAC. Real temporary git repo, scripted replies."""

import json
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output
from orchestrator.baseline import run_b0_task
from orchestrator.checks import final_patch, init_workspace
from orchestrator.scheduler import Budget

GOOD_FIX = "printf 'def add(a, b):\\n    return a + b\\n' > calc.py"
AGENT = {"system_template": "system", "instance_template": "{{task}}"}


def step(command: str) -> dict:
    return make_output("THOUGHT: ok", [{"command": command}])


def run_b0(repo: Path, outputs: list[dict], budget: Budget) -> tuple[dict, Path]:
    out = repo.parent / f"b0_{repo.name}"
    metrics = run_b0_task(
        task_id="t",
        issue="add() subtracts",
        env=LocalEnvironment(cwd=str(repo), timeout=60),
        model=DeterministicModel(outputs=outputs),
        budget=budget,
        out_dir=out,
        agent_config=AGENT,
    )
    return metrics, out


def test_final_patch_keeps_source_and_drops_tests_and_config(repo):
    env = LocalEnvironment(cwd=str(repo), timeout=60)
    base = init_workspace(env)
    env.execute({"command": f"{GOOD_FIX} && echo '# t' >> test_calc.py && echo 'x = 1' > pyproject.toml"})
    patch = final_patch(env, base)
    assert ("calc.py" in patch, "test_calc.py" in patch, "pyproject" in patch) == (True, False, False)


def test_b0_patch_comes_from_the_tree_not_from_its_own_submission_text(repo):
    outputs = [step(GOOD_FIX), step("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo garbage")]
    metrics, out = run_b0(repo, outputs, Budget(max_calls=10, max_activation_calls=10))
    patch = (out / "patch.diff").read_text()
    assert (metrics["exit_status"], metrics["calls"], "+    return a + b" in patch, "garbage" in patch) == (
        "Submitted",
        2,
        True,
        False,
    )


def test_b0_keeps_edits_already_made_when_the_budget_runs_out_before_it_submits(repo):
    metrics, out = run_b0(repo, [step(GOOD_FIX), step("true")], Budget(max_calls=1, max_activation_calls=1))
    assert (metrics["exit_status"], metrics["calls"]) == ("LimitsExceeded", 1)
    assert "+    return a + b" in (out / "patch.diff").read_text()


def test_b0_uses_the_same_hard_budget_and_never_submits_test_changes(repo):
    outputs = [step(f"{GOOD_FIX} && echo '# weaker' >> test_calc.py"), step("true"), step("true")]
    metrics, out = run_b0(repo, outputs, Budget(max_calls=2, max_activation_calls=2))
    assert (metrics["calls"], "test_calc" in (out / "patch.diff").read_text()) == (2, False)
    assert json.loads((out / "metrics.json").read_text())["calls"] == 2


def test_b0_infrastructure_failure_still_writes_metrics(repo):
    metrics, out = run_b0(repo, [make_output("x", [{"raise": RuntimeError("api down")}])], Budget())
    assert (metrics["run_status"], (out / "patch.diff").exists()) == ("infra_failed", True)
    assert "api down" in metrics["errors"][0]
