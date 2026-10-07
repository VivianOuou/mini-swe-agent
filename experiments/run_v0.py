#!/usr/bin/env python3

"""Run EDAC V0 (ArtifactState + four-event rule scheduler) or the B0 baseline (--method b0) on SWE-bench instances.

Both methods run serially, one worker at a time, under the same task-level budget, budget wrapper and patch extraction.

Usage: PYTHONPATH=. python experiments/run_v0.py --slice 0:5 -m openai/gpt-5.4-mini-2026-03-17 --model-class litellm_response \
    --environment-class singularity -o results/edac_v0_mini -c swebench.yaml -c experiments/configs/singularity_testbed.yaml

Infrastructure failures (container build, API errors) are logged to infra_failures.jsonl and the instance gets no
prediction line, so rerunning the command retries exactly those instances. Existing instances are skipped.
"""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import typer
from datasets import load_dataset

import minisweagent
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    DEFAULT_CONFIG_FILE,
    filter_instances,
    get_sb_environment,
)
from minisweagent.utils.serialize import UNSET, recursive_merge
from orchestrator import checks, events
from orchestrator.baseline import run_b0_task
from orchestrator.runner import run_task
from orchestrator.scheduler import Budget

app = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def file_hash(*names: str) -> str:
    return hashlib.sha256(b"".join((ROOT / n).read_bytes() for n in names)).hexdigest()[:16]


def manifest(**settings) -> dict:
    """Everything needed to tie a result directory to the code and settings that produced it. Contains no secrets."""
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "orchestrator", "experiments", "src")),
        "python": sys.version.split()[0],
        "mini_swe_agent": minisweagent.__version__,
        "litellm": version("litellm"),
        "prompts_sha": file_hash("orchestrator/prompts.py"),
        "scheduler_sha": file_hash("orchestrator/scheduler.py", "orchestrator/events.py"),
        "required_checks": list(events.REQUIRED_CHECKS),
        "forbidden_paths": checks.FORBIDDEN_PATH.pattern,
        **settings,
    }


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset"),
    split: str = typer.Option("dev", "--split"),
    slice_spec: str = typer.Option("", "--slice"),
    filter_spec: str = typer.Option("", "--filter"),
    output: Path = typer.Option(..., "-o", "--output"),
    model: str = typer.Option(..., "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    environment_class: str | None = typer.Option(None, "--environment-class"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config"),
    max_tokens: int = typer.Option(1_000_000, "--max-tokens", help="Task-level cap on input+output tokens, all modes"),
    max_calls: int = typer.Option(80, "--max-calls"),
    max_activation_calls: int = typer.Option(20, "--max-activation-calls"),
    max_seconds: int = typer.Option(1800, "--max-seconds"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    method: str = typer.Option("edac", "--method", help="edac or b0"),
) -> None:
    assert method in ("edac", "b0"), method
    output.mkdir(parents=True, exist_ok=True)
    instances = filter_instances(
        list(load_dataset(DATASET_MAPPING.get(subset, subset), split=split)),
        filter_spec=filter_spec,
        slice_spec=slice_spec,
    )
    config = recursive_merge(
        *[get_config_from_spec(spec) for spec in config_spec],
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {"model_name": model, "model_class": model_class or UNSET},
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(
            manifest(
                dataset=DATASET_MAPPING.get(subset, subset),
                split=split,
                slice=slice_spec,
                filter=filter_spec,
                instance_ids=[i["instance_id"] for i in instances],
                method=method,
                model=model,
                model_class=model_class,
                environment_class=environment_class,
                config_specs=config_spec,
                model_kwargs=config.get("model", {}).get("model_kwargs"),
                budget={"max_tokens": max_tokens, "max_calls": max_calls, "max_activation_calls": max_activation_calls},
                max_seconds=max_seconds,
            ),
            indent=2,
        )
    )
    preds = output / "preds.jsonl"
    done = (
        {json.loads(line)["instance_id"] for line in preds.read_text().splitlines()}
        if preds.exists() and not redo_existing
        else set()
    )
    for instance in instances:
        iid = instance["instance_id"]
        if iid in done:
            continue
        started = time.time()
        try:
            env = get_sb_environment(config, instance)
            try:
                budget = Budget(
                    max_tokens=max_tokens,
                    max_calls=max_calls,
                    max_activation_calls=max_activation_calls if method == "edac" else max_calls,
                )
                common = dict(
                    task_id=iid,
                    issue=instance["problem_statement"],
                    env=env,
                    budget=budget,
                    out_dir=output / iid,
                    max_seconds=max_seconds,
                )
                model_obj = get_model(config=config.get("model", {}))
                if method == "edac":
                    metrics = run_task(model=model_obj, **common)
                else:
                    metrics = run_b0_task(model=model_obj, agent_config=config["agent"], **common)
            finally:
                env.cleanup()
        except Exception:
            metrics = {
                "run_status": "infra_failed",
                "calls": 0,
                "tokens": 0,
                "cost_usd": 0.0,
                "errors": [traceback.format_exc()],
            }
        if metrics["run_status"] == "infra_failed":
            with (output / "infra_failures.jsonl").open("a") as f:
                f.write(json.dumps({"instance_id": iid, "errors": metrics["errors"]}) + "\n")
        else:
            with preds.open("a") as f:
                patch = (output / iid / "patch.diff").read_text()
                f.write(json.dumps({"instance_id": iid, "model_name_or_path": model, "model_patch": patch}) + "\n")
        typer.echo(
            f"{iid}: {metrics['run_status']} | calls {metrics['calls']} | tokens {metrics['tokens']} | ${metrics['cost_usd']:.3f} | {time.time() - started:.0f}s"
        )


if __name__ == "__main__":
    app()
