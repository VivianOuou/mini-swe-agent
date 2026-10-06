"""Stage 0, container half: no model. Checks checkout, diff extraction/re-apply, visible checks and log saving in a real
SWE-bench container, then feeds the real results through ArtifactState/events/scheduler.

Usage: python experiments/stage0_container.py [--instance sqlfluff__sqlfluff-1517] [--env docker|singularity]
"""

import json
from pathlib import Path

import typer
from datasets import load_dataset

from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import DEFAULT_CONFIG_FILE, get_sb_environment
from minisweagent.utils.serialize import recursive_merge
from orchestrator.artifacts import ArtifactState, Evidence, Hypothesis, Patch, TestResult
from orchestrator.events import EventQueue, derive_event
from orchestrator.scheduler import Budget, Scheduler

app = typer.Typer(add_completion=False)
HEREDOC = "STAGE0_PATCH_EOF"


@app.command()
def main(
    instance: str = "sqlfluff__sqlfluff-1517",
    env_class: str = typer.Option("singularity", "--env"),
    out: Path = Path("results/stage0"),
):
    row = next(r for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="dev") if r["instance_id"] == instance)
    config = get_config_from_spec(str(DEFAULT_CONFIG_FILE))
    if env_class == "singularity":
        config = recursive_merge(config, get_config_from_spec("experiments/configs/singularity_testbed.yaml"))
    config["environment"] |= {"environment_class": env_class, "timeout": 300}
    env = get_sb_environment(config, row)
    out = out / instance
    (out / "checks").mkdir(parents=True, exist_ok=True)
    for f in ["events.jsonl", "decisions.jsonl"]:
        (out / f).unlink(missing_ok=True)

    def sh(command: str) -> str:
        res = env.execute({"command": command})
        assert res["returncode"] == 0, f"{command!r} failed: {res}"
        return res["output"]

    def check(name: str, test_file: str) -> tuple[int, str]:
        res = env.execute(
            {"command": f"timeout 240 python -m pytest -x -q {test_file} 2>&1 | tail -40; exit ${{PIPESTATUS[0]}}"}
        )
        (out / "checks" / f"{name}.log").write_text(res["output"])
        return res["returncode"], str(out / "checks" / f"{name}.log")

    results: dict[str, bool] = {}

    results["testbed_env_active"] = sh("python -c 'import pytest, sys; print(sys.prefix)'").strip().endswith("testbed")
    head = sh("git rev-parse HEAD").strip()
    results["checkout_at_base_commit"] = head == row["base_commit"]
    results["clean_tree"] = sh("git status --porcelain").strip() == ""

    files = sh("git ls-files").split()
    source = next(f for f in files if f.endswith(".py") and "test" not in f and f.startswith("src/"))
    repro_test, regression_test = [f for f in files if f.startswith("test/") and f.endswith("_test.py")][:2]

    baseline = {n: check(f"base_{n}", t)[0] for n, t in [("repro", repro_test), ("regression", regression_test)]}
    results["baseline_checks_pass_on_clean_base"] = all(c == 0 for c in baseline.values())

    sh(f"echo '# stage0 probe' >> {source}")
    patch = sh("git diff")
    (out / "patch.diff").write_text(patch)
    results["diff_extracted"] = "+# stage0 probe" in patch
    sh("git checkout -- .")
    sh(f"git apply <<'{HEREDOC}'\n{patch}{HEREDOC}")
    results["diff_reapplies_identically"] = sh("git diff") == patch

    state = ArtifactState(task_id=instance, issue=row["problem_statement"][:200], base_commit=head)
    queue, scheduler = EventQueue(out / "events.jsonl"), Scheduler(Budget(), log_path=out / "decisions.jsonl")

    def step() -> str:
        event = derive_event(state)
        if event is None:
            return "none"
        queue.push(event)
        return f"{event.type.value}->{scheduler.decide(queue.pop(state), state).reason}"

    trace = [step()]
    log_path = str(out / "checks" / "explore.log")
    (out / "checks" / "explore.log").write_text(sh(f"git ls-files {source}"))
    state.commit(Evidence("e1", log_path, "r0"))
    state.commit(Hypothesis("h1", "probe", ["e1"], [source]))
    trace.append(step())

    state.commit(Patch("p1", "h1", "r0", patch, ["e1"]), repo_revision="r1")
    trace.append(step())
    codes = {}
    for scope, test_file in [("repro", repro_test), ("regression", regression_test)]:
        code, path = check(f"p1_{scope}", test_file)
        codes[scope] = code
        state.commit(TestResult(f"t_p1_{scope}", "p1", "r1", scope, test_file, code, path))
    trace.append(step())
    results["good_patch_checks_pass"] = all(c == 0 for c in codes.values())

    sh("git checkout -- .")
    sh(f"echo 'raise RuntimeError(\"stage0 deliberate break\")' >> {source}")
    bad_patch = sh("git diff")
    state.commit(Patch("p2", "h1", "r1", bad_patch, ["e1"]), repo_revision="r2")
    code, path = check("p2_repro", repro_test)
    state.commit(TestResult("t_p2_repro", "p2", "r2", "repro", repro_test, code, path))
    trace.append(step())
    results["bad_patch_check_fails"] = code != 0
    results["bad_patch_fails_for_the_right_reason"] = "stage0 deliberate break" in Path(path).read_text()

    sh("git checkout -- .")
    state.save(out / "state.json")
    env.cleanup()
    results["state_roundtrip"] = ArtifactState.load(out / "state.json") == state
    results["trace_is_expected"] = trace == [
        "NeedEvidence->NeedEvidence",
        "ReadyToPatch->ReadyToPatch",
        "none",
        "Solved->solved",
        "PatchFailed->PatchFailed",
    ]
    results["events_logged"] = len((out / "events.jsonl").read_text().splitlines()) == 4
    (out / "stage0_result.json").write_text(
        json.dumps({"results": results, "trace": trace, "exit_codes": codes}, indent=2)
    )
    for name, ok in results.items():
        print(("PASS" if ok else "FAIL"), name)
    print("trace:", " | ".join(trace))
    raise typer.Exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    app()
