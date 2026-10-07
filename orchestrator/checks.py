"""Environment-side helpers: real git diffs anchored to the base commit, patch re-application, frozen check files and the
visible acceptance checks.

Everything here talks to the environment through env.execute, so it works for docker, singularity and local alike.
"""

import re
import shlex
from pathlib import Path

from orchestrator.artifacts import ArtifactState, TestResult

FORBIDDEN_PATH = re.compile(
    r"(^|/)tests?/|_test\.py$|(^|/)test_[^/]*\.py$|(^|/)conftest\.py$|(^|/)(setup\.py|setup\.cfg|pyproject\.toml|tox\.ini)$"
)
SCRATCH_NAME = re.compile(r"(^|/)(repro|reproduce|reproduction|debug|scratch|tmp)[^/]*\.(py|sh|txt)$")
TEST_FILE_FILTER = r"(^|/)test_[^/]*\.py$|_tests?\.py$"
INVALID_REPRO_CODES = (0, 124, 126, 127, -1)
"""Exit codes that cannot demonstrate the issue: success, timeout, command not executable/found, environment timeout."""
SETUP_ERROR = re.compile(r"No such file or directory|can't open file|command not found|No module named pytest")
FAILED_LINE = re.compile(r"^FAILED (\S+)", re.M)
INTENT_TO_ADD = "git add -N -- . ':(exclude)*.pyc'"


def sh(env, command: str, timeout: int | None = None) -> dict:
    return env.execute({"command": command}, timeout=timeout) if timeout else env.execute({"command": command})


def init_workspace(env) -> str:
    """Returns the base commit. .edac/ holds worker scripts and is excluded from every diff."""
    out = sh(
        env,
        "git rev-parse HEAD && mkdir -p .edac && (grep -qx '.edac/' .git/info/exclude || echo '.edac/' >> .git/info/exclude)",
    )
    assert out["returncode"] == 0, out
    return out["output"].splitlines()[0].strip()


def source_diff(env, base: str = "HEAD") -> str:
    """Working tree vs the base commit, so a worker running git commit cannot hide changes from the diff."""
    return sh(env, f"{INTENT_TO_ADD} && git diff {shlex.quote(base)}")["output"]


FINAL_PATCH_EXCLUDES = [
    "tests/**",
    "test/**",
    "test_*.py",
    "*_test.py",
    "conftest.py",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
]


def final_patch(env, base: str = "HEAD") -> str:
    """The patch-extraction rule shared by every method: working tree vs the base commit, minus tests and config files."""
    excludes = " ".join(shlex.quote(f":(exclude,glob)**/{g}") for g in FINAL_PATCH_EXCLUDES)
    return sh(env, f"{INTENT_TO_ADD} && git diff {shlex.quote(base)} -- . {excludes}")["output"]


def changed_files(env, base: str = "HEAD") -> list[str]:
    return sh(env, f"{INTENT_TO_ADD} && git diff {shlex.quote(base)} --name-only")["output"].split()


def reset_to_base(env, base: str = "HEAD") -> None:
    out = sh(env, f"git reset -q --hard {shlex.quote(base)} && git clean -fdq")
    assert out["returncode"] == 0, out


def restore_head(env, base: str) -> bool:
    """Moves HEAD back to base if the worker committed, keeping the working tree. Returns True if HEAD had moved."""
    moved = sh(env, "git rev-parse HEAD")["output"].strip() != base
    if moved:
        sh(env, f"git reset -q --mixed {shlex.quote(base)}")
    return moved


def quarantine_strays(env, scratch_only: bool) -> list[str]:
    """Moves untracked, non-ignored files into .edac/stray/. In Patch mode only scratch/test-looking files move,
    because new source files are a legitimate part of a patch."""
    moved = []
    for f in sh(env, "git ls-files --others --exclude-standard")["output"].splitlines():
        if (
            f.endswith(".pyc")
            or "__pycache__/" in f
            or (scratch_only and not (FORBIDDEN_PATH.search(f) or SCRATCH_NAME.search(f)))
        ):
            continue
        q = shlex.quote(f)
        sh(env, f'mkdir -p "$(dirname .edac/stray/{q})" && mv -- {q} .edac/stray/{q}')
        moved.append(f)
    return moved


def apply_diff(env, diff: str) -> bool:
    """Applies onto the current (clean) tree. git apply is atomic: on failure the tree is unchanged."""
    delimiter = "EDAC_DIFF_EOF"
    while delimiter in diff:
        delimiter += "_X"
    return sh(env, f"git apply --whitespace=nowarn <<'{delimiter}'\n{diff}{delimiter}")["returncode"] == 0


def _write_file(env, path: str, text: str) -> None:
    delimiter = "EDAC_FILE_EOF"
    while delimiter in text:
        delimiter += "_X"
    q = shlex.quote(path)
    sh(env, f"mkdir -p \"$(dirname {q})\" && cat > {q} <<'{delimiter}'\n{text.rstrip(chr(10))}\n{delimiter}")


def snapshot_edac(env) -> dict[str, str]:
    """Text of every worker-written file under .edac/ (except handoff.json and stray/) at the moment the repro is frozen."""
    names = sh(env, "find .edac -type f ! -name handoff.json ! -path '.edac/stray/*' -size -200k | sort")[
        "output"
    ].splitlines()
    return {n: sh(env, f"cat -- {shlex.quote(n)}")["output"] for n in names}


def restore_edac(env, snapshot: dict[str, str]) -> list[str]:
    """Rewrites any frozen file that was edited or deleted. Returns the paths that had been changed."""
    changed = []
    for path, text in snapshot.items():
        now = sh(env, f"cat -- {shlex.quote(path)}")
        if now["returncode"] != 0 or now["output"].rstrip("\n") != text.rstrip("\n"):
            changed.append(path)
            _write_file(env, path, text)
    return changed


def run_check(env, command: str, log_path: Path, timeout: int = 180, tail: int = 80) -> int:
    """Runs a check with a hard timeout and stores the tail of the real output. Returns the real exit code."""
    quoted = shlex.quote(command)
    script = f"(command -v timeout >/dev/null && exec timeout {timeout} bash -c {quoted} || exec bash -c {quoted}) 2>&1 | tail -{tail}; exit ${{PIPESTATUS[0]}}"
    res = sh(env, f"bash -c {shlex.quote(script)}", timeout=timeout + 30)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(res["output"] + (f"\n[{res['exception_info']}]" if res.get("exception_info") else ""))
    return res["returncode"]


def select_regression_command(env, files: list[str], python: str = "python") -> str | None:
    """Existing visible test modules that mention a changed source module (at most 3). None if nothing matches."""
    modules = sorted({Path(f).stem for f in files if not FORBIDDEN_PATH.search(f) and len(Path(f).stem) >= 3})
    if not modules:
        return None
    pattern = "|".join(re.escape(m) for m in modules)
    out = sh(
        env,
        f"git ls-files | grep -E {shlex.quote(TEST_FILE_FILTER)} | xargs grep -liE {shlex.quote(pattern)} 2>/dev/null | head -3",
    )["output"].split()
    return (
        f"{python} -m pytest -q --tb=no -rf -p no:cacheprovider {' '.join(shlex.quote(f) for f in out)}"
        if out
        else None
    )


def regression_failed(code: int, failed: set[str], base_code: int, base_failed: set[str]) -> bool:
    """A regression is a failure that the clean base does not have. Compares failing test ids, not just exit codes."""
    if code == 0:
        return False
    if base_code == 0:
        return True
    return bool(failed - base_failed) or code != base_code


def run_visible_checks(
    env,
    state: ArtifactState,
    patch_id: str,
    log_dir: Path,
    *,
    base: str,
    frozen: dict[str, str],
    python: str = "python",
    timeout: int = 180,
):
    """Runs repro and regression on the current tree. Returns (results, n_check_batches_used, tampered_paths, vacuous).

    Frozen repro files are restored first, so a worker cannot weaken the gate by editing them. Regression compares the
    set of failing tests with the clean base, so only failures the base does not have count against the patch.
    """
    results, batches = [], 0
    n = len(state.test_results)
    tampered = restore_edac(env, frozen)

    def record(scope: str, command: str, code: int, log: Path) -> None:
        results.append(
            TestResult(f"t{n + len(results) + 1}", patch_id, state.repo_revision, scope, command, code, str(log))
        )

    repro = state.checks["repro"]
    log = log_dir / f"{patch_id}_repro.log"
    record("repro", repro, run_check(env, repro, log, timeout), log)
    batches += 1

    regression = select_regression_command(env, changed_files(env, base), python)
    log = log_dir / f"{patch_id}_regression.log"
    if regression is None:
        log.write_text("no existing test module mentions the changed modules; the regression gate is vacuous\n")
        record("regression", "(no matching tests)", 0, log)
        return results, batches, tampered, True
    code = run_check(env, regression, log, timeout, tail=400)
    batches += 1
    if code == 0:
        record("regression", regression, 0, log)
        return results, batches, tampered, False
    diff = source_diff(env, base)
    reset_to_base(env, base)
    base_log = log_dir / f"{patch_id}_regression_base.log"
    base_code = run_check(env, regression, base_log, timeout, tail=400)
    batches += 1
    if not apply_diff(env, diff):
        raise RuntimeError("failed to re-apply the candidate patch after the baseline run")
    if regression_failed(
        code, set(FAILED_LINE.findall(log.read_text())), base_code, set(FAILED_LINE.findall(base_log.read_text()))
    ):
        record("regression", regression, code, log)
    else:
        record("regression", regression + "  # same failures at base, not counted", 0, log)
    return results, batches, tampered, False
