"""ArtifactState: the single JSON-serializable task state. Records are append-only; commit() validates references."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Evidence:
    id: str
    source: str
    """Path of the raw file/command log this observation came from."""
    revision: str
    """Repo revision the observation was made at."""
    summary: str = ""


@dataclass
class Hypothesis:
    id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    status: str = "proposed"


@dataclass
class Patch:
    id: str
    hypothesis_id: str
    base_revision: str
    diff: str
    evidence_ids: list[str] = field(default_factory=list)
    applied: bool = True


@dataclass
class TestResult:
    __test__ = False
    id: str
    patch_id: str
    tested_revision: str
    scope: str
    command: str
    exit_code: int
    log_path: str


@dataclass
class ArtifactState:
    task_id: str
    issue: str
    base_commit: str
    repo_revision: str = "r0"
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    active_hypothesis_id: str = ""
    active_patch_id: str = ""
    state_version: int = 0
    budget_used: dict = field(default_factory=dict)
    run_status: str = "active"
    """One of active, local_success, budget_exhausted, stalled, infra_failed."""

    def get(self, records: list, id: str):
        return next((r for r in records if r.id == id), None)

    def commit(self, *records, repo_revision: str = "") -> int:
        """Validate and append records, optionally move to a new repo revision. Returns the new state_version."""
        ids = {r.id for r in self.evidence}
        for r in records:
            if isinstance(r, Evidence):
                ids.add(r.id)
            elif isinstance(r, Hypothesis | Patch):
                if missing := set(r.evidence_ids) - ids:
                    raise ValueError(f"{r.id} cites unknown evidence {sorted(missing)}")
                if isinstance(r, Patch) and r.base_revision != self.repo_revision:
                    raise ValueError(f"{r.id} is based on {r.base_revision}, current revision is {self.repo_revision}")
            elif isinstance(r, TestResult):
                if r.tested_revision != self.repo_revision:
                    raise ValueError(f"{r.id} tested {r.tested_revision}, current revision is {self.repo_revision}")
                if self.get(self.patches, r.patch_id) is None:
                    raise ValueError(f"{r.id} tests unknown patch {r.patch_id}")
        for r in records:
            match r:
                case Evidence():
                    self.evidence.append(r)
                case Hypothesis():
                    self.hypotheses.append(r)
                    self.active_hypothesis_id = r.id
                case Patch():
                    self.patches.append(r)
                    self.active_patch_id = r.id
                case TestResult():
                    self.test_results.append(r)
        self.repo_revision = repo_revision or self.repo_revision
        self.state_version += 1
        return self.state_version

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ArtifactState":
        data = json.loads(path.read_text())
        for key, record_cls in [
            ("hypotheses", Hypothesis),
            ("evidence", Evidence),
            ("patches", Patch),
            ("test_results", TestResult),
        ]:
            data[key] = [record_cls(**r) for r in data[key]]
        return cls(**data)
