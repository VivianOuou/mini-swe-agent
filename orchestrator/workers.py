"""Worker-side plumbing: a model wrapper that charges the task-level budget on every call, whatever the mode."""

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


def bind_handoff(data: dict, state: ArtifactState, commands: list[str], env, traj_ref: str) -> list:
    """Turns an Explore/Diagnose handoff into Evidence + Hypothesis records. Every claim is checked against what
    really happened: cited commands must have been executed (commands that write or submit the handoff itself do not
    count, otherwise any text could vouch for itself), target files must exist, cites must resolve."""
    records, new_ids = [], []
    for i, item in enumerate(data.get("evidence") or []):
        cmd = str(item.get("cmd", "")).strip() if isinstance(item, dict) else ""
        hit = next(
            (
                k
                for k in reversed(range(len(commands)))
                if len(cmd) >= 6 and cmd in commands[k] and "handoff.json" not in commands[k]
            ),
            None,
        )
        if hit is None:
            raise HandoffError(f"evidence[{i}].cmd was not among the commands you executed: {cmd!r}")
        records.append(
            Evidence(
                f"e{len(state.evidence) + i + 1}",
                f"{traj_ref}#cmd{hit}",
                state.repo_revision,
                str(item.get("note", "")),
            )
        )
        new_ids.append(records[-1].id)
    hyp = data.get("hypothesis")
    if not isinstance(hyp, dict) or not str(hyp.get("text", "")).strip():
        raise HandoffError('handoff needs a "hypothesis" object with non-empty "text"')
    known = {e.id for e in state.evidence} | set(new_ids)
    cites = []
    for c in hyp.get("cites") or []:
        cid = new_ids[c] if isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(new_ids) else c
        if cid not in known:
            raise HandoffError(f"hypothesis cites unknown evidence {c!r}")
        cites.append(cid)
    targets = [str(f) for f in hyp.get("target_files") or []]
    for f in targets:
        if FORBIDDEN_PATH.search(f):
            raise HandoffError(f"target file is a test/config file and cannot be patched: {f}")
        if sh(env, f"test -f {shlex.quote(f)}")["returncode"] != 0:
            raise HandoffError(f"target file does not exist in the repository: {f}")
    records.append(Hypothesis(f"h{len(state.hypotheses) + 1}", str(hyp["text"]).strip(), cites, targets))
    return records


REPAIR_PREFIX = """\
<repair>
Your previous handoff was rejected by the system: {{repair_error}}
Previous handoff:
{{repair_raw}}
You may run more commands if needed, then hand off again. Commands from your previous session still count as executed.
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
    repair: tuple[str, str] | None = None,
) -> dict:
    """One bounded activation of the frozen worker. Charges the shared budget through `model`."""
    model.begin_activation(cap)
    template = render_instance_template(mode, has_repro)
    agent = DefaultAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=(REPAIR_PREFIX + template) if repair else template,
        step_limit=0,
        cost_limit=0,
        output_path=traj_path,
    )
    info = agent.run(
        task, state_view=view, repair_error=repair[0] if repair else "", repair_raw=repair[1] if repair else ""
    )
    commands = [a["command"] for m in agent.messages for a in m.get("extra", {}).get("actions", [])]
    return {"exit_status": info.get("exit_status", ""), "submission": info.get("submission", ""), "commands": commands}
