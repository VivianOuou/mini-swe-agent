"""Worker-side plumbing: a model wrapper that charges the task-level budget on every call, whatever the mode."""

import difflib
import json
import re
import shlex
from pathlib import Path

from minisweagent import Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded
from orchestrator.artifacts import ArtifactState, Evidence, Hypothesis
from orchestrator.checks import FORBIDDEN_PATH, sh
from orchestrator.prompts import SYSTEM_TEMPLATE, render_instance_template
from orchestrator.scheduler import Budget, Mode

SOURCE_EDIT_MODES = {Mode.PATCH}
"""Only Patch may change production code; the adapter checks the real diff against this."""


def _approx_tokens(messages: list[dict]) -> int:
    """Rough chars/4 size of what is actually sent to or received from the provider (the bookkeeping `extra` is not)."""
    return len(json.dumps([{k: v for k, v in m.items() if k != "extra"} for m in messages], default=str)) // 4


def message_tokens(message: dict, prompt: list[dict] | None = None) -> tuple[int, bool]:
    """Returns (tokens, estimated). Chat API: extra.response.usage (prompt/completion_tokens). Responses API: top-level
    usage (input/output_tokens). Without usage, falls back to an estimate of the prompt plus the reply."""
    usage = message.get("usage") or (message.get("extra", {}).get("response") or {}).get("usage")
    if isinstance(usage, dict):
        tokens_in = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        return tokens_in + usage.get("output_tokens", usage.get("completion_tokens", 0)), False
    return _approx_tokens(prompt or []) + _approx_tokens([message]), True


def message_cost(message: dict) -> float:
    return message.get("extra", {}).get("cost", 0.0)


class BudgetedModel:
    """Wraps any mini Model. The same Budget instance must be shared across all activations of one task."""

    def __init__(self, model: Model, budget: Budget):
        self.model = model
        self.budget = budget
        self.activation_calls = 0
        self.activation_cap: int | None = None

    def begin_activation(self, cap: int | None = None) -> None:
        self.activation_calls, self.activation_cap = 0, cap

    def query(self, messages: list[dict], **kwargs) -> dict:
        if not self.budget.can_call(_approx_tokens(messages), self.activation_calls, self.activation_cap):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.activation_calls += 1
        try:
            message = self.model.query(messages, **kwargs)
        except FormatError as e:
            self.budget.charge(*message_tokens(e.messages[0], messages), message_cost(e.messages[0]))
            raise
        self.budget.charge(*message_tokens(message, messages), message_cost(message))
        return message

    def __getattr__(self, name: str):
        return getattr(self.model, name)


class HandoffError(ValueError):
    """The worker's handoff is malformed or cites things that did not happen."""


def parse_handoff(text: str) -> dict:
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise HandoffError(f"handoff is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise HandoffError("handoff must be a JSON object")
    return data


SIMILARITY = 0.85
"""Minimum similarity to accept a paraphrased citation. Measured on real mini runs: misquoted-but-real commands scored
0.95+, never-executed or half-invented ones 0.64 and below, fabricated controls 0.52 and below."""


def _bare(cmd: str) -> str:
    return re.sub(r"^cd\s+\S+\s*&&\s*", "", cmd.strip())


def executed_index(cmd: str, commands: list[str]) -> int | None:
    """Index of the executed command that vouches for `cmd`, or None.

    First every non-trivial && / ; part of `cmd` must occur verbatim in a command that really ran; failing that, `cmd`
    must be near-identical (SIMILARITY) to one that did. A command that merely writes or submits the handoff file never
    counts, and a part nobody ran never passes.
    """
    real = [(k, c) for k, c in enumerate(commands) if "handoff.json" not in c]
    parts = [p for p in (x.strip() for x in re.split(r"&&|;", cmd)) if p and not re.fullmatch(r"cd\s+\S+", p)]
    long_parts = [p for p in parts if len(p) >= 4]
    if not long_parts:
        return next((k for k, c in reversed(real) if c.strip() == cmd and cmd), None)
    hits = [next((k for k, c in reversed(real) if p in c), None) for p in long_parts]
    if None not in hits:
        return max(hits)
    ratio, best = max(
        ((difflib.SequenceMatcher(None, _bare(cmd), _bare(c)).ratio(), k) for k, c in real), default=(0, None)
    )
    return best if ratio >= SIMILARITY else None


def bind_handoff(data: dict, state: ArtifactState, commands: list[str], env, traj_ref: str) -> tuple[list, list[str]]:
    """Turns an Explore/Diagnose handoff into Evidence + Hypothesis records. Returns (records, dropped notes).

    Only claims that can be checked against what really happened enter the state: an evidence item whose command did
    not run is dropped (and reported), never recorded, and cites that point at dropped or unknown evidence are dropped
    too. A hypothesis left without verified evidence simply fails the ReadyToPatch gate. Structural problems (bad
    shape, missing/forbidden/nonexistent target files) still reject the handoff so it can be repaired."""
    records, dropped = [], []
    new_ids: list[str | None] = []
    for i, item in enumerate(data.get("evidence") or []):
        cmd = str(item.get("cmd", "")).strip() if isinstance(item, dict) else ""
        if (hit := executed_index(cmd, commands)) is None:
            dropped.append(f"evidence[{i}] cites a command that was not executed: {cmd[:160]!r}")
            new_ids.append(None)
            continue
        records.append(
            Evidence(
                f"e{len(state.evidence) + len(records) + 1}",
                f"{traj_ref}#cmd{hit}",
                state.repo_revision,
                str(item.get("note", "")),
            )
        )
        new_ids.append(records[-1].id)
    hyp = data.get("hypothesis")
    if not isinstance(hyp, dict) or not str(hyp.get("text", "")).strip():
        raise HandoffError('handoff needs a "hypothesis" object with non-empty "text"')
    known = {e.id for e in state.evidence} | {i for i in new_ids if i}
    cites = []
    for c in hyp.get("cites") or []:
        cid = new_ids[c] if isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(new_ids) else c
        if cid in known:
            cites.append(cid)
        else:
            dropped.append(f"hypothesis cite {c!r} points at evidence that is unverified or unknown")
    targets = [str(f) for f in hyp.get("target_files") or []]
    for f in targets:
        if FORBIDDEN_PATH.search(f):
            raise HandoffError(f"target file is a test/config file and cannot be patched: {f}")
        if sh(env, f"test -f {shlex.quote(f)}")["returncode"] != 0:
            raise HandoffError(f"target file does not exist in the repository: {f}")
    records.append(Hypothesis(f"h{len(state.hypotheses) + 1}", str(hyp["text"]).strip(), cites, targets))
    return records, dropped


REPAIR_PREFIX = """\
<repair>
Your previous handoff was rejected by the system: {{repair_error}}
Do NOT investigate further and do not run other commands. Only fix the handoff JSON and submit it again with the
two-step sequence (you have at most 3 tool calls). When citing evidence, copy a command EXACTLY (or a prefix of it) from
this list of the commands you actually executed:
{{repair_commands}}
Previous handoff:
{{repair_raw}}
</repair>

"""


def run_activation(
    *,
    mode: Mode,
    model: BudgetedModel,
    env,
    task: str,
    view: str,
    has_repro: bool,
    traj_path: Path,
    cap: int | None = None,
    repair: tuple[str, str, list[str]] | None = None,
) -> dict:
    """One bounded activation of the frozen worker. Charges the shared budget through `model`."""
    model.begin_activation(cap)
    template = render_instance_template(mode, has_repro, cap or model.budget.max_activation_calls)
    agent = DefaultAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=(REPAIR_PREFIX + template) if repair else template,
        step_limit=0,
        cost_limit=0,
        output_path=traj_path,
    )
    listed = (
        "\n".join(f"  [{i}] {c[:300]}" for i, c in enumerate(repair[2]) if "handoff.json" not in c) if repair else ""
    )
    info = agent.run(
        task,
        state_view=view,
        repair_error=repair[0] if repair else "",
        repair_raw=repair[1] if repair else "",
        repair_commands=listed,
    )
    commands = [a["command"] for m in agent.messages for a in m.get("extra", {}).get("actions", [])]
    return {"exit_status": info.get("exit_status", ""), "submission": info.get("submission", ""), "commands": commands}
