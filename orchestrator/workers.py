"""Worker-side plumbing: a model wrapper that charges the task-level budget on every call, whatever the mode."""

from minisweagent import Model
from minisweagent.exceptions import FormatError, LimitsExceeded
from orchestrator.scheduler import Budget, Mode

SOURCE_EDIT_MODES = {Mode.PATCH}
"""Only Patch may change production code; the adapter checks the real diff against this."""


def message_tokens(message: dict) -> tuple[int, bool]:
    """Returns (tokens, estimated). Chat API: extra.response.usage (prompt/completion_tokens). Responses API: top-level
    usage (input/output_tokens). Falls back to a chars/4 estimate when the provider reports no usage."""
    usage = message.get("usage") or (message.get("extra", {}).get("response") or {}).get("usage")
    if isinstance(usage, dict):
        tokens_in = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        return tokens_in + usage.get("output_tokens", usage.get("completion_tokens", 0)), False
    return len(str(message.get("content") or "")) // 4, True


class BudgetedModel:
    """Wraps any mini Model. The same Budget instance must be shared across all activations of one task."""

    def __init__(self, model: Model, budget: Budget, reserve_tokens: int = 4096):
        self.model = model
        self.budget = budget
        self.reserve_tokens = reserve_tokens
        self.activation_calls = 0

    def begin_activation(self) -> None:
        self.activation_calls = 0

    def query(self, messages: list[dict], **kwargs) -> dict:
        if not self.budget.can_call(self.reserve_tokens, self.activation_calls):
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
            self.budget.charge(*message_tokens(e.messages[0]))
            raise
        self.budget.charge(*message_tokens(message))
        return message

    def __getattr__(self, name: str):
        return getattr(self.model, name)
