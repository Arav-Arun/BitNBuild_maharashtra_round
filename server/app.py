"""FastAPI adapter for the Black Box service layer."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from blackbox.api.errors import ApiError
from blackbox.api.service import BlackBoxService
from server import models as m

logger = logging.getLogger("blackbox.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(__import__("os").environ.get("DATA_DIR", "data"))
    has_recordings = data_dir.is_dir() and any(
        (child / "blackbox.db").is_file() for child in data_dir.iterdir()
    )
    service = BlackBoxService(data_root=data_dir, static_bundle=not has_recordings)
    app.state.service = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="Black Box API", version=m.CONTRACT_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in __import__("os")
        .environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        .split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Last-Event-ID"],
)


def service(request: Request) -> BlackBoxService:
    return request.app.state.service


def require_recordings(svc: BlackBoxService) -> None:
    if svc.static_bundle:
        raise ApiError(
            "unavailable",
            "The recorded API bundle has no dataset attached.",
            hint="The web client will show its labelled read-only demo fixtures.",
        )


def _error(error: ApiError, request_id: str | None = None) -> Response:
    body = m.ErrorResponse(error=error.body(request_id))
    return Response(body.model_dump_json(), status_code=error.status, media_type="application/json")


@app.middleware("http")
async def request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, error: ApiError):
    return _error(error, getattr(request.state, "request_id", None))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, error: RequestValidationError):
    issues = [
        m.ValidationIssue(loc=list(i["loc"]), message=i["msg"], type=i["type"])
        for i in error.errors()
    ]
    body = m.ErrorResponse(
        error=m.ErrorBody(
            code="validation_error",
            message="The request did not match the API contract.",
            status=422,
            hint="Check the field locations in issues.",
            issues=issues,
            context=None,
            request_id=getattr(request.state, "request_id", None),
        )
    )
    return Response(body.model_dump_json(), status_code=422, media_type="application/json")


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, error: Exception):
    logger.exception("Unhandled API error", exc_info=error)
    return _error(
        ApiError("internal_error", "The API could not complete this request."),
        getattr(request.state, "request_id", None),
    )


@app.get("/health", response_model=m.AppHealth)
def health(request: Request):
    svc = service(request)
    result = svc.health()
    if not svc.static_bundle:
        return result
    capabilities = result.capabilities.model_copy(
        update={
            "fork": False,
            "verify": False,
            "export_test": False,
            "label": False,
            "ingest_otlp": False,
        }
    )
    return result.model_copy(
        update={
            "static_bundle": True,
            "notes": [
                *result.notes,
                "No recorded database is attached; the web app uses read-only fixture data.",
            ],
            "capabilities": capabilities,
        }
    )


@app.get("/agents", response_model=m.AgentList)
def agents(request: Request):
    svc = service(request)
    require_recordings(svc)
    return svc.list_agents()


@app.get("/runs", response_model=m.RunList)
def runs(request: Request, query: m.RunListQuery = __import__("fastapi").Depends()):
    svc = service(request)
    require_recordings(svc)
    return svc.list_runs(query)


@app.get("/failure-groups", response_model=m.FailureGroupList)
def failure_groups(request: Request, agent: str | None = None, split: str | None = None):
    svc = service(request)
    require_recordings(svc)
    return svc.failure_groups(agent, split)


@app.get("/runs/{run_id}", response_model=m.RunDetail)
def run_detail(request: Request, run_id: str, blind: bool = False):
    svc = service(request)
    require_recordings(svc)
    return svc.run_detail(run_id, blind=blind)


@app.get("/runs/{run_id}/steps/{addr}", response_model=m.StepDetail)
def step(request: Request, run_id: str, addr: str):
    return service(request).step(run_id, addr)


@app.get("/runs/{run_id}/provenance", response_model=m.ValueProvenance)
def provenance(request: Request, run_id: str, addr: str, pointer: str):
    return service(request).provenance(run_id, addr, pointer)


@app.get("/runs/{run_id}/diagnosis", response_model=m.Diagnosis)
def diagnosis(request: Request, run_id: str, refresh: bool = False):
    return service(request).diagnosis(run_id, refresh=refresh)


@app.post("/replay/predict", response_model=m.ReplayPrediction)
def replay_predict(request: Request, body: m.ForkRequest):
    svc = service(request)
    require_recordings(svc)
    return svc.forks.predict(body)


@app.post("/forks", response_model=m.ForkCreated, status_code=202)
async def create_fork(request: Request, body: m.ForkRequest):
    svc = service(request)
    require_recordings(svc)
    return await svc.forks.create(body)


@app.get("/forks/{fork_id}/stream")
async def fork_stream(request: Request, fork_id: str):
    svc = service(request)
    # Validate before returning headers; streaming errors cannot become JSON responses.
    svc.forks._meta(fork_id)

    async def events():
        async for item in svc.forks.stream(fork_id):
            if await request.is_disconnected():
                break
            event = item if isinstance(item, dict) else item.model_dump(mode="json")
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {json.dumps(event['data'], separators=(',', ':'))}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/forks/{fork_id}", response_model=m.ForkSummary)
def fork_summary(request: Request, fork_id: str):
    return service(request).forks.summary(fork_id)


@app.get("/runs/{run_id}/forks", response_model=m.ForkTimeline)
def fork_timeline(request: Request, run_id: str):
    return service(request).forks.timeline(run_id)


@app.post("/runs/{run_id}/verify", response_model=m.VerifyJob, status_code=202)
async def verify(request: Request, run_id: str, body: m.VerifyRequest):
    svc = service(request)
    require_recordings(svc)
    return await svc.forks.start_verify(run_id, body)


@app.get("/jobs/{job_id}", response_model=m.VerifyJob)
def verify_job(request: Request, job_id: str):
    return service(request).forks.get_job(job_id)


@app.get("/diff", response_model=m.DiffResponse)
def diff(request: Request, a: str, b: str):
    svc = service(request)
    left, right = svc.run_detail(a), svc.run_detail(b)
    if svc.agent_of(a) is not svc.agent_of(b):
        raise ApiError("validation_error", "Runs from different agents cannot be aligned.")
    nearest = None
    if left.run.status == "failed" and right.run.status == "passed":
        nearest = m.NearestPassingLink(
            run_id=b,
            similarity=1.0,
            reason="Compared directly",
        )
    return __import__("blackbox.api.reader", fromlist=["build_diff"]).build_diff(
        left, right, invalidated=set(), nearest=nearest
    )


@app.get("/runs/{run_id}/twin", response_model=m.NearestTwin)
def twin(request: Request, run_id: str):
    return service(request).twin(run_id)


@app.get("/eval", response_model=m.EvalResponse)
def evaluation(request: Request):
    svc = service(request)
    path = svc.eval_dir / "api.json"
    if path.is_file():
        return m.EvalResponse.model_validate_json(path.read_text(encoding="utf-8"))
    try:
        from blackbox.api.eval_adapter import build_eval

        return build_eval(svc)
    except FileNotFoundError:
        pass
    fixture = Path(__file__).parent.parent / "web" / "mocks" / "eval.json"
    if fixture.is_file():
        return m.EvalResponse.model_validate_json(fixture.read_text(encoding="utf-8"))
    raise ApiError("unavailable", "Evaluation report has not been generated.")


@app.get("/runs/{run_id}/report", response_model=m.CrashReport)
def crash_report(request: Request, run_id: str):
    svc = service(request)
    detail = svc.run_detail(run_id)
    if detail.run.status != "failed":
        raise ApiError("not_applicable", "Crash reports are only available for failed runs.")
    try:
        diagnosis = svc.diagnosis(run_id)
    except ApiError as error:
        if error.status >= 500:
            diagnosis = None
        else:
            raise
    suspect = diagnosis.responsible_addr if diagnosis else None
    reason = detail.run.checker_reason or "The run failed its task checks."
    events = [
        m.ReportEvent(
            seq=s.seq,
            addr=s.addr,
            text=f"{s.name}: {('error: ' + s.error.type) if s.error else ('visible failure' if diagnosis and s.addr == diagnosis.visible_failure_addr else 'recorded step')}",
            tag="root"
            if s.addr == suspect
            else "symptom"
            if diagnosis and s.addr == diagnosis.visible_failure_addr
            else None,
        )
        for s in detail.steps
    ]
    cause = m.ReportClaim(
        text=f"The leading candidate is {suspect}. {reason}" if suspect else reason,
        cites=[m.Citation(addr=suspect, json_pointer=None)] if suspect else [],
    )
    return m.CrashReport(
        run_id=run_id,
        title=f"Failure report · {detail.run.task}",
        generated_at=datetime.now(UTC),
        synopsis=f"{detail.run.agent} failed: {reason}",
        sequence_of_events=events,
        probable_cause=cause,
        contributing_factors=[],
        findings=(
            [
                m.ReportClaim(text=reason.text, cites=[reason.citation])
                for reason in diagnosis.reasons
            ]
            if diagnosis
            else []
        ),
        recommended_fix=None,
        verification=(
            [m.ReportClaim(text=diagnosis.verification.explanation, cites=[])]
            if diagnosis and diagnosis.verification
            else None
        ),
        fixture=False,
    )


@app.get("/runs/{run_id}/report.md")
def crash_report_markdown(request: Request, run_id: str):
    report = crash_report(request, run_id)
    lines = [f"# {report.title}", "", report.synopsis, "", "## Event sequence", ""]
    lines.extend(f"- `{e.addr}` — {e.text}" for e in report.sequence_of_events)
    if report.probable_cause:
        lines.extend(["", "## Probable cause", "", report.probable_cause.text])
    if report.findings:
        lines.extend(["", "## Evidence", ""])
        lines.extend(f"- {f.text}" for f in report.findings)
    return Response("\n".join(lines) + "\n", media_type="text/markdown")


@app.post("/forks/{fork_id}/export-test", response_model=m.ExportTestResponse)
def export_test(request: Request, fork_id: str, body: m.ExportTestRequest):
    svc = service(request)
    require_recordings(svc)
    from blackbox.export.evidence import export_verified

    return export_verified(svc, fork_id, overwrite=body.overwrite)


@app.get("/labels/queue", response_model=m.LabelQueue)
def label_queue(
    request: Request, annotator: str = "anonymous", limit: int = Query(50, ge=1, le=200)
):
    svc = service(request)
    require_recordings(svc)
    runs = [r for r in svc._all_rows() if r["outcome"] == "failed" and r["origin"] != "fork"]
    labelled = {
        r["run_id"]
        for a in svc.agents.values()
        for r in a.database.query("SELECT run_id FROM human_labels WHERE annotator=?", (annotator,))
    }
    items = []
    for row in runs:
        if row["run_id"] in labelled:
            continue
        items.append(
            m.LabelTask(
                run_id=row["run_id"],
                agent=row["agent"],
                task=row["task"],
                checker_reason=row["checker_reason"],
                steps=row["n_steps"],
                labelled_by_you=False,
            )
        )
        if len(items) >= limit:
            break
    target = min(50, len(runs))
    return m.LabelQueue(
        progress=m.LabelProgress(
            labelled=len(labelled), target=target, remaining=max(0, target - len(labelled))
        ),
        items=items,
    )


@app.post("/labels", response_model=m.LabelResponse, status_code=201)
def save_label(request: Request, body: m.LabelRequest):
    svc = service(request)
    require_recordings(svc)
    detail = svc.run_detail(body.run_id)
    if body.root_addr and body.root_addr not in {s.addr for s in detail.steps}:
        raise ApiError("validation_error", f"Run has no step {body.root_addr!r}.")
    agent = svc.agent_of(body.run_id)
    label_id = uuid.uuid4().hex
    created = datetime.now(UTC)
    try:
        agent.database.execute(
            "INSERT INTO human_labels(label_id,run_id,annotator,root_addr,certainty,notes,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                label_id,
                body.run_id,
                body.annotator,
                body.root_addr,
                body.certainty,
                body.notes,
                created.isoformat(),
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ApiError("conflict", "This annotator already labelled the run.") from error
    all_labels = [
        row
        for data in svc.agents.values()
        for row in data.database.query("SELECT run_id, annotator, root_addr FROM human_labels")
    ]
    by_annotator: dict[str, dict[str, str]] = {}
    for row in all_labels:
        by_annotator.setdefault(row["annotator"], {})[row["run_id"]] = (
            row["root_addr"] or "__no_responsible_step__"
        )
    annotators = sorted(by_annotator)
    pairwise: list[tuple[float, int]] = []
    double_labelled_runs: set[str] = set()
    for index, left in enumerate(annotators):
        for right in annotators[index + 1 :]:
            left_labels, right_labels = by_annotator[left], by_annotator[right]
            overlap = set(left_labels) & set(right_labels)
            if not overlap:
                continue
            double_labelled_runs.update(overlap)
            observed = sum(left_labels[run] == right_labels[run] for run in overlap) / len(overlap)
            categories = set(left_labels[run] for run in overlap) | set(
                right_labels[run] for run in overlap
            )
            expected = sum(
                sum(left_labels[run] == category for run in overlap)
                * sum(right_labels[run] == category for run in overlap)
                for category in categories
            ) / (len(overlap) ** 2)
            kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
            pairwise.append((kappa, len(overlap)))
    weighted = sum(value * count for value, count in pairwise)
    overlap_n = sum(count for _, count in pairwise)
    agreement = m.KappaStatus(
        kappa=weighted / overlap_n if overlap_n else None,
        n=len(double_labelled_runs),
        annotators=len(annotators),
    )
    total = sum(r["outcome"] == "failed" for r in svc._all_rows())
    count = len(
        {
            r["run_id"]
            for a in svc.agents.values()
            for r in a.database.query(
                "SELECT run_id FROM human_labels WHERE annotator=?", (body.annotator,)
            )
        }
    )
    return m.LabelResponse(
        label_id=label_id,
        run_id=body.run_id,
        annotator=body.annotator,
        root_addr=body.root_addr,
        created_at=created,
        progress=m.LabelProgress(
            labelled=count, target=min(50, total), remaining=max(0, min(50, total) - count)
        ),
        agreement=agreement,
    )


@app.post("/v1/traces", response_model=m.OtlpIngestAck)
async def ingest_otlp(request: Request):
    # OTLP is accepted in the standard path. Each resource/scope span is turned into an
    # imported, non-replayable trace by the dedicated importer.
    svc = service(request)
    require_recordings(svc)
    payload = await request.json()
    from blackbox.api.otlp import ingest

    return ingest(svc, payload)
