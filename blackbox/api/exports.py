"""HTTP and MCP response adapter for the shared offline regression exporter."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from datetime import UTC, datetime

from blackbox.api.errors import ApiError
from blackbox.api.service import BlackBoxService
from blackbox.export.regression import ExportError, export
from server import models as m


def export_verified(
    service: BlackBoxService, fork_id: str, *, overwrite: bool = False
) -> m.ExportTestResponse:
    fork = service.forks.summary(fork_id)
    if fork.verdict != "VERIFIED" or not fork.edits:
        raise ApiError("not_verified", "Only a VERIFIED intervention can be exported.")
    agent = service.agent_of(fork.base_run_id)
    target = service.data_root / "exports" / f"fork-{fork_id}"
    if target.exists():
        if not overwrite:
            raise ApiError(
                "conflict",
                "This fork was already exported.",
                hint="Send overwrite=true to regenerate it.",
            )
        shutil.rmtree(target)
    try:
        result = export(agent.recorder, fork_id, target, adapter_args=agent.replay_args)
    except ExportError as error:
        raise ApiError("not_applicable", str(error)) from error

    files = [
        m.ExportedFile(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
        )
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]
    manifest_path = result.fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_path = {file.path: file.sha256 for file in files}
    return m.ExportTestResponse(
        export_id=result.export_id,
        fork_id=fork_id,
        verdict="VERIFIED",
        test_path=str(result.test_path),
        fixture_dir=str(result.fixture_dir),
        test_sha256=by_path[str(result.test_path)],
        fixture_sha256=by_path[str(manifest_path)],
        files=files,
        invalidation_cone=manifest["cone"],
        run_command=f"uv run --locked --extra dev python {shlex.quote(str(result.test_path))}",
        created_at=datetime.now(UTC),
    )
