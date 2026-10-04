"""The path of damage: from a suspect, forward along provenance edges to the final answer.

Every step the suspect's output reaches is tagged:

- ``root``        the suspect itself;
- ``symptom``     its output differs from the same-task passing twin (or, with no
                  comparable twin, it looks anomalous on its own);
- ``unaffected``  it consumed the damage but produced the twin's output anyway
                  (early cutoff), or looks healthy.

``basis`` says which comparison produced the tags, so the UI never presents an
anomaly guess as if it were a twin comparison.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from blackbox.ml.dataset import Trace
from blackbox.ml.features import output_payload
from blackbox.recorder import canonical_json


def _children(trace: Trace) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    known = set(trace.addrs)
    for src, dst, _ in trace.edges:
        if src in known and dst in known and src != dst:
            children[src].add(dst)
    return children


def _same(a: Any, b: Any) -> bool:
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def damage_path(
    trace: Trace,
    suspect: str,
    *,
    twin: Trace | None = None,
    anomaly: dict[str, float] | None = None,
) -> dict[str, Any]:
    children = _children(trace)
    parent: dict[str, str] = {}
    reached = {suspect}
    queue = deque([suspect])
    while queue:
        node = queue.popleft()
        for child in sorted(children.get(node, ())):
            if child not in reached:
                reached.add(child)
                parent[child] = node
                queue.append(child)

    ordered = sorted((s for s in trace.steps if s.addr in reached), key=lambda s: s.seq)
    twin_steps = {s.addr: s for s in twin.steps} if twin is not None else {}
    basis = "twin" if twin is not None else "anomaly"
    nodes = []
    for step in ordered:
        if step.addr == suspect:
            tag = "root"
        elif step.addr in twin_steps:
            same = _same(output_payload(step)[0], output_payload(twin_steps[step.addr])[0])
            tag = "unaffected" if same else "symptom"
        else:
            tag = "symptom" if (anomaly or {}).get(step.addr, 0.0) >= 1.0 else "unaffected"
        nodes.append({"addr": step.addr, "seq": step.seq, "kind": step.kind, "tag": tag})

    # A suspect that produced the twin's own output did not start the damage below it:
    # those symptoms came from elsewhere, and the UI must not draw them as its fault.
    origin = twin_steps.get(suspect)
    suspect_step = next((s for s in ordered if s.addr == suspect), None)
    suspect_matches_twin = (
        origin is not None
        and suspect_step is not None
        and _same(output_payload(suspect_step)[0], output_payload(origin)[0])
    )
    final = max(trace.steps, key=lambda s: s.seq).addr if trace.steps else None
    path: list[str] = []
    if final in reached:
        node = final
        while node != suspect:
            path.append(node)
            node = parent[node]
        path.append(suspect)
        path.reverse()
    return {
        "suspect": suspect,
        "basis": basis,
        "suspect_matches_twin": suspect_matches_twin,
        "nodes": nodes,
        "edges": sorted(
            {(src, dst) for src, dst, _ in trace.edges if src in reached and dst in reached}
        ),
        "path_to_final": path,
        "reaches_final": bool(path),
        "reached": len(nodes) - 1,
        "symptoms": sum(node["tag"] == "symptom" for node in nodes),
    }
