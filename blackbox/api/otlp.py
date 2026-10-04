"""Small OTLP/HTTP JSON importer for GenAI spans. Imported traces are read-only."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from blackbox.api import index as idx
from blackbox.api.agents import profile
from blackbox.api.reader import RunReader
from blackbox.api.service import AgentData
from blackbox.sdk import Recorder
from server import models as m


def _attrs(items: list[dict[str, Any]]) -> dict[str, Any]:
    def value(v: dict[str, Any]) -> Any:
        if "stringValue" in v:
            return v["stringValue"]
        if "intValue" in v:
            try:
                return int(v["intValue"])
            except (TypeError, ValueError):
                return v["intValue"]
        if "doubleValue" in v:
            return v["doubleValue"]
        if "boolValue" in v:
            return v["boolValue"]
        if "arrayValue" in v:
            return [value(x) for x in v["arrayValue"].get("values", [])]
        if "kvlistValue" in v:
            return {x["key"]: value(x.get("value", {})) for x in v["kvlistValue"].get("values", [])}
        if "bytesValue" in v:
            return v["bytesValue"]
        return None

    return {
        x["key"]: value(x.get("value", {})) for x in items if isinstance(x, dict) and "key" in x
    }


def _time(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1_000_000_000, UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC)


def ingest(service, payload: dict[str, Any]) -> m.OtlpIngestAck:
    accepted: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    rejected = 0
    spans_count = 0
    for resource in payload.get("resourceSpans", []):
        attrs = _attrs(resource.get("resource", {}).get("attributes", []))
        for scope in resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", [])):
            for span in scope.get("spans", []):
                spans_count += 1
                trace_id = span.get("traceId", "")
                if not re.fullmatch(r"[a-fA-F0-9]{32}", trace_id) or not span.get("spanId"):
                    rejected += 1
                    continue
                accepted.setdefault(trace_id.lower(), []).append((span, attrs))
    if not accepted:
        return m.OtlpIngestAck(
            partialSuccess=(
                m.OtlpPartialSuccess(
                    rejectedSpans=rejected, errorMessage="No valid spans were found."
                )
                if rejected
                else None
            ),
            accepted_spans=0,
            runs=[],
            replayable=False,
            note="Provide OTLP/HTTP JSON resourceSpans with traceId and spanId fields.",
        )

    data_dir = service.data_root / "imported"
    agent = service.agents.get("imported")
    if agent is None:
        recorder = Recorder(data_dir, mode=service.mode, settings=service.settings)
        prof = profile("imported")
        agent = AgentData(
            name="imported",
            data_dir=data_dir,
            recorder=recorder,
            profile=prof,
            reader=RunReader(recorder, idx.FAULT_SPECS),
            replay_args={},
        )
        service.agents["imported"] = agent
    refs = []
    for trace_id, trace_spans in accepted.items():
        run_id = f"otlp-{trace_id}"
        if agent.database.one("SELECT run_id FROM runs WHERE run_id=?", (run_id,)):
            rejected += len(trace_spans)
            continue
        trace_spans.sort(
            key=lambda pair: (_time(pair[0].get("startTimeUnixNano")), pair[0].get("spanId", ""))
        )
        started = min(
            (_time(s.get("startTimeUnixNano")) for s, _ in trace_spans), default=datetime.now(UTC)
        )
        ended = max((_time(s.get("endTimeUnixNano")) for s, _ in trace_spans), default=started)
        agent.database.execute(
            "INSERT INTO runs(run_id,parent_run_id,fork_id,agent,agent_version,task_id,model,seed,mode,outcome,score,checker_reason,started_at,ended_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                None,
                None,
                "imported",
                "otlp",
                "OTLP trace",
                None,
                None,
                "recorded",
                "running",
                None,
                None,
                started.isoformat(),
                ended.isoformat(),
            ),
        )
        result = []
        for span, resource_attrs in trace_spans:
            attrs = {**resource_attrs, **_attrs(span.get("attributes", []))}
            opname = str(
                attrs.get("gen_ai.operation.name")
                or attrs.get("openinference.span.kind")
                or "agent step"
            )
            kind = (
                "retrieval"
                if "retriev" in opname.lower()
                else "tool"
                if "tool" in opname.lower()
                else "llm"
            )
            raw_input = attrs.get(
                "gen_ai.input.messages", attrs.get("input.value", attrs.get("input", {}))
            )
            raw_output = attrs.get(
                "gen_ai.output.messages", attrs.get("output.value", attrs.get("output", {}))
            )

            def decode(v):
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except ValueError:
                        return v
                return v

            inp, out = decode(raw_input), decode(raw_output)
            state_ref = agent.store.save_checkpoint({})
            in_ref = agent.store.save_json(inp)
            out_ref = agent.store.save_json(out)
            started_step = _time(span.get("startTimeUnixNano"))
            ended_step = _time(span.get("endTimeUnixNano"))
            error = (
                span.get("status", {}).get("code") == "2" or span.get("status", {}).get("code") == 2
            )
            step_addr = f"{kind}/{re.sub(r'[^A-Za-z0-9_-]+', '-', opname).strip('-') or 'span'}#{len(result) + 1}"
            token_in = attrs.get("gen_ai.usage.input_tokens")
            token_out = attrs.get("gen_ai.usage.output_tokens")
            agent.database.execute(
                """INSERT INTO steps(run_id,addr,seq,kind,name,agent_role,request_key,input_hash,output_hash,reasoning_hash,state_before,state_after,reads_json,writes_json,cache_status,tokens_in,tokens_out,tokens_cached,latency_ms,finish_reason,error_type,retries) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    step_addr,
                    len(result) + 1,
                    kind,
                    opname,
                    None,
                    None,
                    in_ref,
                    out_ref,
                    None,
                    state_ref,
                    state_ref,
                    "[]",
                    "[]",
                    "live",
                    token_in,
                    token_out,
                    0,
                    max(0, (ended_step - started_step).total_seconds() * 1000),
                    None,
                    "OTelError" if error else None,
                    0,
                ),
            )
            result.append((span, attrs, step_addr, error, inp, out))
        fail = any(item[3] for item in result)
        reason = next(
            (
                str(item[1].get("exception.message"))
                for item in result
                if item[3] and item[1].get("exception.message")
            ),
            None,
        )
        agent.database.execute(
            "UPDATE runs SET outcome=?,score=?,checker_reason=? WHERE run_id=?",
            ("failed" if fail else "passed", 0.0 if fail else 1.0, reason, run_id),
        )
        for index_, item in enumerate(result):
            if index_ + 1 < len(result):
                agent.database.execute(
                    "INSERT OR IGNORE INTO edges(run_id,src_addr,dst_addr,kind) VALUES (?,?,?,'inferred')",
                    (run_id, item[2], result[index_ + 1][2]),
                )
        rows = idx.build_rows(
            agent.database,
            agent.store,
            agent.profile,
            [agent.database.one("SELECT * FROM runs WHERE run_id=?", (run_id,))],
        )
        idx.insert_rows(agent.database, rows)
        refs.append(m.OtlpRunRef(run_id=run_id, spans=len(trace_spans)))
    service._refresh_run_ids()
    service.cache.rows = None
    return m.OtlpIngestAck(
        partialSuccess=m.OtlpPartialSuccess(
            rejectedSpans=rejected,
            errorMessage="Some spans were malformed or duplicate." if rejected else "",
        )
        if rejected
        else None,
        accepted_spans=sum(r.spans for r in refs),
        runs=refs,
        replayable=False,
        note="Imported GenAI trace data is read-only and cannot be replayed.",
    )
