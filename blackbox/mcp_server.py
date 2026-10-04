"""Local MCP tools for recorded diagnosis and verified replay (stdio transport)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from blackbox.api.exports import export_verified
from blackbox.api.service import BlackBoxService
from server import models as contract

mcp = FastMCP(
    "Black Box",
    instructions=(
        "Inspect recorded agent runs, inspect steps, create paired replay interventions, "
        "and export verified fixes as offline pytest checks. Diagnosis is a hypothesis; "
        "only the paired replay verdict establishes verification."
    ),
)
_service: BlackBoxService | None = None


def service() -> BlackBoxService:
    global _service
    if _service is None:
        _service = BlackBoxService(data_root=Path(os.environ.get("DATA_DIR", "data")))
    return _service


@mcp.tool()
def get_suspects(run_id: str) -> dict[str, Any]:
    """Return ranked candidate steps, evidence and abstention status for a failed run."""
    return service().diagnosis(run_id).model_dump(mode="json")


@mcp.tool()
def get_step(run_id: str, addr: str) -> dict[str, Any]:
    """Read the full recorded inputs, outputs, state and provenance metadata for one step."""
    return service().step(run_id, addr).model_dump(mode="json")


@mcp.tool()
async def fork_and_verify(
    run_id: str,
    addr: str,
    kind: str,
    value: Any,
    samples: int = 5,
    known_good: bool = False,
) -> dict[str, Any]:
    """Edit one step, selectively replay it, and compare against an unchanged control."""
    svc = service()
    request = contract.ForkRequest(
        base_run_id=run_id,
        edits=[contract.ForkEdit(addr=addr, kind=kind, value=value, known_good=known_good)],
        mode="cone",
        samples=samples,
        control=True,
        hypothesis="MCP paired intervention",
    )
    created = await svc.forks.create(request)
    await svc.forks.wait(created.fork_id)
    return svc.forks.summary(created.fork_id).model_dump(mode="json")


@mcp.tool()
def export_regression_test(fork_id: str) -> dict[str, Any]:
    """Export a VERIFIED intervention and its offline pytest evidence fixture."""
    return export_verified(service(), fork_id).model_dump(mode="json")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
