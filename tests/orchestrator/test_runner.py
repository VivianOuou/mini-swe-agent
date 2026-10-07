"""End-to-end loop tests on a real temporary git repo with scripted (deterministic) worker replies. No model, no mocks."""

import json
import subprocess
import sys
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output
from orchestrator.runner import run_task
from orchestrator.scheduler import Budget

PY = sys.executable
REPRO = f"PYTHONPATH=. {PY} .edac/repro.py"
GOOD_FIX = "printf 'def add(a, b):\\n    return a + b\\n' > calc.py"
BAD_FIX = "printf 'def add(a, b):\\n    return a * b\\n' > calc.py"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat .edac/handoff.json"


def step(command: str) -> dict:
    return make_output("THOUGHT: ok", [{"command": command}])


def handoff(payload: dict) -> list[dict]:
    return [step(f"cat > .edac/handoff.json <<'HANDOFF_EOF'\n{json.dumps(payload)}\nHANDOFF_EOF"), step(SUBMIT)]


def explore(extra_step: str | None = None, hypothesis: dict | None = None) -> list[dict]:
    hypothesis = hypothesis or {"text": "add subtracts", "target_files": ["calc.py"], "cites": [0]}
    return [
        step("mkdir -p .edac && printf 'from calc import add\\nassert add(1, 2) == 3\\n' > .edac/repro.py"),
        *([step(extra_step)] if extra_step else []),
        step(f"{REPRO}; echo exit=$?"),
        *handoff(
            {"evidence": [{"cmd": REPRO, "note": "add(1, 2) != 3"}], "hypothesis": hypothesis, "repro_cmd": REPRO}
        ),
    ]


def patch(fix: str) -> list[dict]:
    return [step(fix), *handoff({"summary": "fix add", "cites": ["e1"]})]


def run(repo: Path, outputs: list[dict], budget: Budget | None = None, max_seconds: int = 1800) -> tuple[dict, Path]:
    metrics = run_task(
        task_id="t",
        issue="add() subtracts instead of adding",
        env=LocalEnvironment(cwd=str(repo), timeout=60),
        model=DeterministicModel(outputs=outputs),
        budget=budget or Budget(),
        out_dir=repo.parent / f"out_{repo.name}",
        python=PY,
        max_seconds=max_seconds,
    )
    return metrics, repo.parent / f"out_{repo.name}"


def event_types(out: Path) -> list[str]:
    return [json.loads(line)["type"] for line in (out / "events.jsonl").read_text().splitlines()]


def test_explore_then_patch_solves_and_keeps_scratch_files_out_of_the_patch(repo):
    metrics, out = run(repo, explore("echo '# probe' >> calc.py") + patch(GOOD_FIX))
    assert (metrics["run_status"], metrics["activations"], metrics["calls"], metrics["check_batches"]) == (
        "local_success",
        {"Explore": 1, "Patch": 1},
        len(explore("x")) + len(patch(GOOD_FIX)),
        3,
    )
    assert metrics["errors"] == ["Explore modified source files; changes reverted"]
    diff = (out / "patch.diff").read_text()
    assert ("+    return a + b" in diff, "# probe" in diff, ".edac" in diff) == (True, False, False)
    assert event_types(out) == ["NeedEvidence", "ReadyToPatch", "Solved"]
    state = json.loads((out / "state.json").read_text())
    assert [(r["scope"], r["exit_code"], r["tested_revision"]) for r in state["test_results"]] == [
        ("repro", 0, "r1"),
        ("regression", 0, "r1"),
    ]
    assert state["evidence"][-1]["summary"] == "repro fails at base (exit 1)"


def test_failed_patch_goes_through_diagnose_and_a_fresh_patch_from_a_clean_base(repo):
    diagnose = [
        step(f"{REPRO}; echo exit=$?"),
        *handoff(
            {
                "evidence": [{"cmd": REPRO, "note": "still fails with a*b"}],
                "hypothesis": {"text": "must add, not multiply", "target_files": ["calc.py"], "cites": ["e1", 0]},
            }
        ),
    ]
    metrics, out = run(repo, explore() + patch(BAD_FIX) + diagnose + patch(GOOD_FIX))
    assert (metrics["run_status"], metrics["activations"], metrics["patches"]) == (
        "local_success",
        {"Explore": 1, "Patch": 2, "Diagnose": 1},
        2,
    )
    assert event_types(out) == ["NeedEvidence", "ReadyToPatch", "PatchFailed", "ReadyToPatch", "Solved"]
    state = json.loads((out / "state.json").read_text())
    assert [p["base_revision"] for p in state["patches"]] == ["r0", "r2"]
    assert "a * b" not in (out / "patch.diff").read_text()
    assert [d["mode"] for d in map(json.loads, (out / "decisions.jsonl").read_text().splitlines())] == [
        "Explore",
        "Patch",
        "Diagnose",
        "Patch",
        None,
    ]


def test_patch_touching_tests_is_discarded_and_retried(repo):
    metrics, out = run(repo, explore() + patch("echo '# weaker' >> test_calc.py") + patch(GOOD_FIX))
    assert (metrics["run_status"], metrics["patches"], metrics["activations"]["Patch"]) == ("local_success", 1, 2)
    assert metrics["errors"] == ["patch discarded: touched ['test_calc.py']"]
    assert "test_calc.py" not in (out / "patch.diff").read_text()


def test_unverifiable_handoffs_are_repaired_once_then_the_run_stalls(repo):
    bad = [step("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo 'not json'")]
    metrics, out = run(repo, bad * 4)
    assert (metrics["run_status"], metrics["calls"], metrics["repair_activations"], metrics["activations"]) == (
        "stalled",
        4,
        2,
        {"Explore": 2},
    )
    assert len(metrics["errors"]) == 2 and metrics["patches"] == 0


def test_evidence_must_cite_a_command_that_really_ran_and_repro_must_fail_at_base(repo):
    lie = explore(hypothesis=None)[:-2] + handoff(
        {
            "evidence": [{"cmd": "grep -rn secret_answer .", "note": "made up"}],
            "hypothesis": {"text": "x"},
            "repro_cmd": REPRO,
        }
    )
    metrics, _ = run(repo, lie + lie, Budget(max_calls=len(lie) * 2))
    assert any("was not among the commands you executed" in e for e in metrics["errors"])
    passing_repro = [
        step("mkdir -p .edac && printf 'assert True\\n' > .edac/repro.py"),
        step(f"{PY} .edac/repro.py"),
        *handoff(
            {
                "evidence": [{"cmd": f"{PY} .edac/repro.py", "note": "ok"}],
                "hypothesis": {"text": "x"},
                "repro_cmd": f"{PY} .edac/repro.py",
            }
        ),
    ]
    metrics, _ = run(repo, passing_repro * 2, Budget(max_calls=len(passing_repro) * 2))
    assert (
        any("does not reproduce the issue" in e and "exit code 0" in e for e in metrics["errors"])
        and metrics["patches"] == 0
    )


def test_budget_is_a_hard_stop_across_activations(repo):
    metrics, out = run(repo, explore() + patch(GOOD_FIX), Budget(max_calls=3))
    assert (metrics["run_status"], metrics["stop_reason"], metrics["calls"]) == (
        "budget_exhausted",
        "budget_exhausted",
        3,
    )
    assert json.loads((out / "state.json").read_text())["budget_used"]["calls"] == 3


def test_a_worker_cannot_weaken_the_frozen_repro_to_force_solved(repo):
    rewrite = "echo '# noop' >> calc.py && printf 'import sys\\nsys.exit(0)\\n' > .edac/repro.py"
    outputs = explore() + patch(rewrite)
    metrics, out = run(repo, outputs, Budget(max_calls=len(outputs)))
    assert (metrics["local_solved"], metrics["tampered_check_files"]) == (False, 1)
    assert event_types(out) == ["NeedEvidence", "ReadyToPatch", "PatchFailed"]
    assert (repo / ".edac/repro.py").read_text().strip().endswith("assert add(1, 2) == 3")
    assert any("frozen check files" in e for e in metrics["errors"])


def test_git_commit_by_a_worker_cannot_hide_the_fix_from_the_patch(repo):
    commit = f"{GOOD_FIX} && git -c user.email=a@b.c -c user.name=n commit -qam fix"
    metrics, out = run(repo, explore() + patch(commit))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert (metrics["local_solved"], head == json.loads((out / "state.json").read_text())["base_commit"]) == (
        True,
        True,
    )
    assert "+    return a + b" in (out / "patch.diff").read_text()
    assert any("moved HEAD" in e for e in metrics["errors"])


def test_a_regression_the_base_does_not_have_fails_the_patch_even_though_the_base_also_fails_something(repo):
    break_zero = "printf 'def add(a, b):\\n    return 99 if (a, b) == (0, 0) else a + b\\n' > calc.py"
    outputs = explore() + patch(break_zero)
    metrics, out = run(repo, outputs, Budget(max_calls=len(outputs)))
    results = {r["scope"]: r["exit_code"] for r in json.loads((out / "state.json").read_text())["test_results"]}
    assert (results["repro"], results["regression"] != 0, event_types(out)[-1]) == (0, True, "PatchFailed")
    assert "test_other.py::test_zero" in (out / "checks" / "p1_regression.log").read_text()


def test_a_repeated_failure_without_new_evidence_stalls_instead_of_looping(repo):
    again = handoff({"hypothesis": {"text": "try again", "target_files": ["calc.py"], "cites": ["e1"]}})
    metrics, _ = run(repo, explore() + patch(BAD_FIX) + again + patch(BAD_FIX) + again + patch(BAD_FIX))
    assert (metrics["run_status"], metrics["activations"]) == ("stalled", {"Explore": 1, "Patch": 3, "Diagnose": 2})


def test_time_limit_stops_work_before_any_model_call(repo):
    metrics, _ = run(repo, explore(), max_seconds=-1)
    assert (metrics["stop_reason"], metrics["run_status"], metrics["calls"]) == (
        "time_exhausted",
        "budget_exhausted",
        0,
    )


def test_an_infrastructure_failure_still_saves_state_and_metrics(repo):
    metrics, out = run(repo, [make_output("x", [{"raise": RuntimeError("api down")}])])
    assert (metrics["run_status"], (out / "state.json").exists(), (out / "metrics.json").exists()) == (
        "infra_failed",
        True,
        True,
    )
    assert "api down" in metrics["errors"][0]


def test_a_rerun_does_not_append_to_the_previous_runs_logs(repo):
    run(repo, explore() + patch(GOOD_FIX))
    subprocess.run("git checkout -q -- . && git clean -fdq", shell=True, cwd=repo, check=True)
    _, out = run(repo, [make_output("x", [{"raise": RuntimeError("api down")}])])
    assert event_types(out) == ["NeedEvidence"]


def test_stray_scratch_files_are_moved_aside_not_shipped_or_deleted(repo):
    metrics, out = run(
        repo, explore() + patch(f"{GOOD_FIX} && echo 'print(1)' > reproduce.py && echo x > test_scratch.py")
    )
    diff = (out / "patch.diff").read_text()
    assert (metrics["local_solved"], "reproduce.py" in diff, "test_scratch" in diff) == (True, False, False)
    assert (repo / ".edac/stray/reproduce.py").exists()


def test_a_repro_outside_dot_edac_is_rejected_not_mistaken_for_a_reproduction(repo):
    cmd = f"PYTHONPATH=. {PY} repro.py"
    outputs = [
        step("printf 'from calc import add\\nassert add(1, 2) == 3\\n' > repro.py"),
        step(f"{cmd}; echo $?"),
        *handoff(
            {
                "evidence": [{"cmd": cmd, "note": "fails"}],
                "hypothesis": {"text": "x", "target_files": ["calc.py"], "cites": [0]},
                "repro_cmd": cmd,
            }
        ),
    ]
    metrics, _ = run(repo, outputs * 2, Budget(max_calls=len(outputs) * 2))
    assert any("does not reproduce the issue" in e for e in metrics["errors"]) and metrics["patches"] == 0
    assert any("moved stray files" in e for e in metrics["errors"])


def test_diagnose_cannot_edit_production_code_while_a_patch_is_applied(repo):
    diagnose = [step("echo '# diag' >> calc.py"), *handoff({"hypothesis": {"text": "x"}})]
    outputs = explore() + patch(BAD_FIX) + diagnose
    metrics, _ = run(repo, outputs, Budget(max_calls=len(outputs)))
    assert "Diagnose modified source files; changes reverted" in metrics["errors"]
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a * b\n"


def test_a_lightly_paraphrased_citation_is_accepted_but_an_unexecuted_part_is_not(repo):
    ran = "cd . && pwd && cat calc.py"

    def with_citation(cmd: str) -> list[dict]:
        return [
            step("mkdir -p .edac && printf 'from calc import add\\nassert add(1, 2) == 3\\n' > .edac/repro.py"),
            step(ran),
            *handoff(
                {
                    "evidence": [{"cmd": cmd, "note": "reads calc"}],
                    "hypothesis": {"text": "add subtracts", "target_files": ["calc.py"], "cites": [0]},
                    "repro_cmd": REPRO,
                }
            ),
        ]

    metrics, out = run(repo, with_citation("cd . && cat calc.py") + patch(GOOD_FIX))
    assert (metrics["run_status"], metrics["errors"]) == ("local_success", [])
    bad = with_citation("cat calc.py && grep -rn never_ran .")
    metrics, _ = run(repo, bad * 2, Budget(max_calls=len(bad) * 2))
    assert any("was not among the commands you executed" in e for e in metrics["errors"]) and metrics["patches"] == 0


def test_repair_prompt_lists_the_real_commands_and_forbids_new_investigation(repo):
    bad = [step("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo 'not json'")]
    _, out = run(repo, bad * 4)
    first_user_message = json.loads((out / "trajectories" / "01_Explore_repair.traj.json").read_text())["messages"][1][
        "content"
    ]
    assert (
        "Do NOT investigate further" in first_user_message
        and "[0] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in first_user_message
    )
