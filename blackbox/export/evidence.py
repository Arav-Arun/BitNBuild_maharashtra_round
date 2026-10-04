"""Export a verified replay as a small, network-free evidence regression."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from blackbox.api.errors import ApiError
from blackbox.api.reader import descendants
from blackbox.api.service import BlackBoxService
from server import models as m


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_verified(
    service: BlackBoxService, fork_id: str, *, overwrite: bool = False
) -> m.ExportTestResponse:
    fork = service.forks.summary(fork_id)
    if fork.verdict != "VERIFIED" or not fork.edits or not fork.edited_runs:
        raise ApiError(
            "not_verified", "Only a VERIFIED edited fork has regression evidence to export."
        )
    agent = service.agent_of(fork.base_run_id)
    base = service.run_detail(fork.base_run_id)
    fixed_ref = next((item for item in fork.edited_runs if item.passed), None)
    if fixed_ref is None:
        raise ApiError("not_verified", "The verified fork has no passing sample to export.")
    fixed = service.run_detail(fixed_ref.run_id)
    edited = sorted({edit.addr for edit in fork.edits})
    cone = sorted(descendants(base.edges, edited))
    base_steps = {step.addr: step for step in base.steps}
    fixed_steps = {step.addr: step for step in fixed.steps}
    fixture = {
        "format": "blackbox-evidence-regression-v1",
        "fork_id": fork_id,
        "base_run_id": base.run.run_id,
        "fixed_run_id": fixed.run.run_id,
        "base_outcome": base.run.status,
        "fixed_outcome": fixed.run.status,
        "edited": edited,
        "invalidation_cone": cone,
        "base_steps": {
            addr: step.hashes.model_dump(mode="json") for addr, step in base_steps.items()
        },
        "fixed_steps": {
            addr: step.hashes.model_dump(mode="json") for addr, step in fixed_steps.items()
        },
        "pass_rate": fork.fix.model_dump(mode="json") if fork.fix else None,
        "control_rate": fork.control_result.model_dump(mode="json")
        if fork.control_result
        else None,
    }
    test_source = '''"""Generated offline check of a verified Black Box intervention."""
import json
from pathlib import Path

FIXTURE = json.loads((Path(__file__).parent / "fixture.json").read_text())


def test_verified_fix_and_invalidation_cone_are_preserved():
    assert FIXTURE["base_outcome"] == "failed"
    assert FIXTURE["fixed_outcome"] == "passed"
    cone = set(FIXTURE["invalidation_cone"])
    edited = set(FIXTURE["edited"])
    assert edited <= cone
    before, after = FIXTURE["base_steps"], FIXTURE["fixed_steps"]
    assert set(before) == set(after)
    # State hashes include the full branch state, so a downstream state can differ
    # even when a cached step's own call stayed identical. Compare call evidence
    # (inputs, outputs, reasoning and request key) when checking invalidation.
    call_fields = ("input", "output", "reasoning", "request_key")
    changed_calls = {
        addr
        for addr in before
        if any(before[addr][field] != after[addr][field] for field in call_fields)
    }
    assert changed_calls <= cone
    assert changed_calls & edited
'''
    export_id = uuid.uuid4().hex
    # One export directory per fork, so `overwrite` decides whether a re-export replaces it.
    target = service.data_root / "exports" / f"fork-{fork_id}"
    if target.exists() and not overwrite:
        raise ApiError(
            "conflict",
            f"This fork was already exported to {target}.",
            hint="Send overwrite=true to regenerate it.",
            context={"fixture_dir": str(target)},
        )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    fixture_path = target / "fixture.json"
    # A distinct module name per export lets pytest collect several exports in one run.
    test_path = target / f"test_fork_{fork_id[:12]}.py"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_path.write_text(test_source, encoding="utf-8")
    fixture_hash, test_hash = _digest(fixture_path), _digest(test_path)
    agent.database.execute(
        "INSERT INTO regression_exports(export_id,fork_id,test_hash,fixture_hash,created_at) VALUES (?,?,?,?,?)",
        (export_id, fork_id, test_hash, fixture_hash, datetime.now(UTC).isoformat()),
    )
    files = [
        m.ExportedFile(path=str(test_path), sha256=test_hash, bytes=test_path.stat().st_size),
        m.ExportedFile(
            path=str(fixture_path), sha256=fixture_hash, bytes=fixture_path.stat().st_size
        ),
    ]
    return m.ExportTestResponse(
        export_id=export_id,
        fork_id=fork_id,
        verdict="VERIFIED",
        test_path=str(test_path),
        fixture_dir=str(target),
        test_sha256=test_hash,
        fixture_sha256=fixture_hash,
        files=files,
        invalidation_cone=cone,
        run_command=f"uv run --extra dev pytest {test_path}",
        created_at=datetime.now(UTC),
    )
