"""Frozen mode prompts and the worker-visible state view. Only the issue and the committed ArtifactState reach a worker."""

from pathlib import Path

from jinja2 import StrictUndefined, Template

from orchestrator.artifacts import ArtifactState
from orchestrator.scheduler import Mode

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."

_COMMON = """\
<instructions>
You work in /testbed, a git checkout of a repository, using bash tool calls. Every command runs in a fresh subshell.
Put any script you write (reproduction scripts, probes) under /testbed/.edac/ - that directory is never part of the patch.
Never modify tests or configuration files (setup.py, setup.cfg, pyproject.toml, tox.ini).
Never run git commit, checkout, reset, stash or branch commands: the system tracks changes against the original commit itself.
Files under .edac/ that the acceptance checks depend on are frozen by the system; edits to them are undone.
Your response MUST contain at least one bash tool call. When you are done, you MUST hand off with the exact two-step
sequence below, as SEPARATE commands. After the second command you cannot do anything else.

Step 1: write the handoff file (JSON, schema below):
  cat > .edac/handoff.json <<'HANDOFF_EOF'
  ...json...
  HANDOFF_EOF
Step 2: submit it:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat .edac/handoff.json

{{ mode_rules }}
</instructions>
"""

_EVIDENCE_RULES = """\
"evidence": every item is {"cmd": "<a command you actually ran, copied from your own tool calls>", "note": "<what it showed>"}.
  Only commands you really executed count; a cmd that was not executed invalidates the handoff.
"hypothesis": {"text": "<a testable explanation>", "target_files": ["<existing repo paths to change>"],
  "cites": [<indexes into your "evidence" list, or existing evidence ids such as "e2">]}.
  If you cannot name target files backed by evidence, leave "target_files" and "cites" empty and list what is missing in "unresolved".
"unresolved": ["<specific missing facts>"]"""

_RULES = {
    Mode.EXPLORE: """\
MODE: EXPLORE. Read code and run probes to find where and why the issue happens. Do NOT edit repository source files.
{% if not has_repro %}You MUST write a reproduction script and give its command as "repro_cmd". It must FAIL (non-zero exit)
on the unmodified repository and exit 0 once the issue is fixed. A repro that already passes is rejected.
{% endif %}Handoff JSON:
{"evidence": [...], "hypothesis": {...}, "unresolved": [...]{% if not has_repro %}, "repro_cmd": "python .edac/repro.py"{% endif %}}
"""
    + _EVIDENCE_RULES,
    Mode.DIAGNOSE: """\
MODE: DIAGNOSE. A candidate patch failed a visible check. Read the actual diff and the real check output in <state>,
investigate why, and revise the hypothesis. Do NOT edit repository source files here.
Always give a NEW hypothesis: either a revised, evidence-backed one that names target files, or one with empty
"target_files" if you need more evidence (then say what in "unresolved").
Handoff JSON:
{"evidence": [...], "hypothesis": {...}, "unresolved": [...]}
"""
    + _EVIDENCE_RULES,
    Mode.PATCH: """\
MODE: PATCH. Implement the fix for the active hypothesis in <state> by editing only the hypothesis' target source files
(or other non-test source files if truly necessary). The tree starts clean. Do not decide on your own that the fix works:
the system runs the acceptance checks after you hand off. You may run your repro script to guide yourself.
Handoff JSON: {"summary": "<one sentence>", "cites": ["<existing evidence ids you relied on>"]}
""",
}

INSTANCE_TEMPLATE = (
    """\
<issue>
{{task}}
</issue>

<state>
{{state_view}}
</state>

"""
    + _COMMON
)


def render_instance_template(mode: Mode, has_repro: bool) -> str:
    """Instance template with the mode rules baked in; {{task}} and {{state_view}} stay for the agent to fill."""
    rules = Template(_RULES[mode], undefined=StrictUndefined).render(has_repro=has_repro)
    return INSTANCE_TEMPLATE.replace("{{ mode_rules }}", rules)


def _tail(path: str, n: int = 40) -> str:
    p = Path(path)
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else "(log unavailable)"


def state_view(state: ArtifactState, mode: Mode, current_diff: str = "") -> str:
    """Local, traceable view. Contains observations and real check output, never dataset gold/test fields."""
    out = [f"mode: {mode.value} | repo revision: {state.repo_revision} | state version: {state.state_version}"]
    out.append("checks: " + (", ".join(f"{k}: {v}" for k, v in state.checks.items()) or "none yet"))
    if state.evidence:
        out.append("evidence:")
        out += [f"  {e.id} [{e.source}] {e.summary}" for e in state.evidence]
    hyp = state.get(state.hypotheses, state.active_hypothesis_id)
    if hyp:
        out.append(
            f"active hypothesis {hyp.id}: {hyp.text} | target files: {hyp.target_files} | cites: {hyp.evidence_ids}"
        )
    patch = state.get(state.patches, state.active_patch_id)
    if patch and mode == Mode.DIAGNOSE:
        out.append(f"latest patch {patch.id} (applied={patch.applied}) diff:\n{current_diff or patch.diff}")
        for r in (r for r in state.test_results if r.patch_id == patch.id and r.tested_revision == state.repo_revision):
            out.append(f"check {r.scope} (exit {r.exit_code}) `{r.command}` output tail:\n{_tail(r.log_path)}")
    if state.unresolved_questions:
        out.append("unresolved: " + "; ".join(state.unresolved_questions))
    if state.errors:
        out.append("system notes (most recent last):\n" + "\n".join(f"  - {e[:300]}" for e in state.errors[-3:]))
    return "\n".join(out)
