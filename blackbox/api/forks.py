"""Forks, their live event streams, the fork timeline and verification jobs.

A fork is one paired K-sample replay of a base run. The replay engine runs it in the
background; its raw events are turned into contract SSE events as they happen, fanned out
to every subscriber, and saved, so a finished stream can be replayed byte for byte.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from blackbox.api.errors import ApiError, classify_replay_error, not_found
from blackbox.api.reader import descendants, normalise_statuses, predict_replay
from blackbox.replay import Edit, ReplayEngine
from server import models as m

if TYPE_CHECKING:
    from blackbox.api.service import BlackBoxService

logger = logging.getLogger("blackbox.api.forks")

MAX_CONCURRENT_FORKS = 2
STANDIN_PREFIXES = ("tripcrew-fixture-", "hoprag-reader-fixture-", "hoprag-lexical-")


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ForkJob:
    fork_id: str
    request: m.ForkRequest
    agent: str
    branch_name: str
    created_at: datetime
    events: list[Any] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    status: m.ForkStatus = "queued"
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


@dataclass
class VerifyState:
    job: m.VerifyJob
    task: asyncio.Task | None = None


def edit_summary(edits: list[m.ForkEdit]) -> str | None:
    if not edits:
        return None
    parts = []
    for edit in edits:
        value = json.dumps(edit.value, sort_keys=True)
        value = value if len(value) <= 60 else value[:57] + "…"
        parts.append(f"{edit.kind} {edit.addr} {value}")
    return "; ".join(parts)


class ForkManager:
    def __init__(self, service: BlackBoxService) -> None:
        self.service = service
        self.jobs: dict[str, ForkJob] = {}
        self.verify_jobs: dict[str, VerifyState] = {}
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_FORKS)
        return self._semaphore

    # ------------------------------------------------------------------
    # Prediction and creation
    # ------------------------------------------------------------------

    def _blocked(
        self, request: m.ForkRequest, prediction: m.ReplayPrediction
    ) -> m.ErrorBody | None:
        service = self.service
        if service.static_bundle:
            return ApiError(
                "unsupported",
                "This is the static recorded bundle; forks need the API server.",
                hint="Run `make dev-api` locally, or open the deployed API.",
            ).body()
        agent = service.agent_of(request.base_run_id)
        if agent.adapter() is None:
            return ApiError(
                "unsupported",
                f"Runs of {agent.name} cannot be replayed here: {agent.adapter_error}",
            ).body()
        run = service.run_row(request.base_run_id)
        detail = service.run_detail(request.base_run_id)
        kinds = {step.addr: step.kind for step in detail.steps}
        live_llm = [
            p.addr
            for p in prediction.per_step
            if kinds.get(p.addr) == "llm" and p.expected == "invalidated"
        ]
        live_llm += [e.addr for e in request.edits if e.kind in {"patch_prompt", "swap_model"}]
        if not live_llm:
            return None
        model = run["model"] or ""
        if service.mode == "recorded":
            return ApiError(
                "live_call_refused",
                f"This fork re-runs {len(set(live_llm))} model call(s), and the server is in "
                "RECORDED mode, which only serves recorded responses.",
                hint="Run the API with MODE=offline (local stand-ins) or MODE=live.",
                context={"steps": sorted(set(live_llm))},
            ).body()
        if not model.startswith(STANDIN_PREFIXES) and not service.settings.groq_api_key:
            return ApiError(
                "live_call_refused",
                "This run was recorded with a hosted model and no API key is configured.",
                hint="Set GROQ_API_KEY to replay it, or fork a run recorded offline.",
            ).body()
        return None

    def predict(self, request: m.ForkRequest) -> m.ReplayPrediction:
        detail = self.service.run_detail(request.base_run_id)
        addrs = {step.addr for step in detail.steps}
        unknown = sorted({e.addr for e in request.edits} - addrs)
        if request.resample_from and request.resample_from not in addrs:
            unknown.append(request.resample_from)
        if unknown:
            raise ApiError(
                "validation_error",
                f"The base run has no step {', '.join(unknown)}.",
                context={"unknown": unknown},
            )
        prediction = predict_replay(detail, request)
        return prediction.model_copy(update={"blocked": self._blocked(request, prediction)})

    async def create(self, request: m.ForkRequest) -> m.ForkCreated:
        prediction = self.predict(request)
        if prediction.blocked is not None:
            body = prediction.blocked
            raise ApiError(body.code, body.message, hint=body.hint, context=body.context)
        agent = self.service.agent_of(request.base_run_id)
        fork_id = uuid.uuid4().hex
        branch = request.branch_name or f"fork-{fork_id[:8]}"
        created = _now()
        agent.database.execute(
            """
            INSERT INTO fork_meta(fork_id, base_run_id, status, hypothesis, control,
                                  request_json, created_at)
            VALUES (?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                fork_id,
                request.base_run_id,
                request.hypothesis,
                int(request.control),
                request.model_dump_json(),
                created.isoformat(),
            ),
        )
        job = ForkJob(fork_id, request, agent.name, branch, created)
        self.jobs[fork_id] = job
        job.task = asyncio.create_task(self._run(job))
        return m.ForkCreated(
            fork_id=fork_id,
            base_run_id=request.base_run_id,
            branch_name=branch,
            status="queued",
            stream_url=f"/forks/{fork_id}/stream",
            prediction=prediction,
            created_at=created,
        )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _emit(self, job: ForkJob, event: Any) -> None:
        job.events.append(event)
        for queue in job.subscribers:
            queue.put_nowait(event)

    def _event(self, job: ForkJob, kind: str, data: Any) -> Any:
        number = len(job.events) + 1
        if kind == "step":
            return m.StepEvent(id=number, event="step", data=data)
        if kind == "outcome":
            return m.OutcomeEvent(id=number, event="outcome", data=data)
        if kind == "summary":
            return m.SummaryEvent(id=number, event="summary", data=data)
        return m.StreamErrorEvent(id=number, event="error", data=data)

    async def _run(self, job: ForkJob) -> None:
        service = self.service
        agent = service.agents[job.agent]
        request = job.request
        base_id = request.base_run_id
        base_steps = {
            row["addr"]: row
            for row in agent.database.query(
                "SELECT addr, request_key FROM steps WHERE run_id = ?", (base_id,)
            )
        }
        edited = {edit.addr for edit in request.edits}
        roots = edited or ({request.resample_from} if request.resample_from else set())
        invalidated = descendants(agent.reader.edges(base_id), roots)

        def on_event(raw: dict[str, Any]) -> None:
            data = raw["data"]
            if raw["event"] == "step":
                addr = data["addr"]
                phase = data["phase"]
                status = data.get("cache_status")
                ms = data.get("ms")
                tokens = data.get("tokens")
                if phase == "done":
                    has_call = (
                        base_steps[addr]["request_key"] is not None
                        if addr in base_steps
                        else status != "live"
                    )
                    if not has_call and addr not in invalidated:
                        status = "cached"
                    if addr in edited and has_call and status == "live":
                        status = "edited"
                    tokens = 0 if status == "cached" else tokens
                    ms = round(ms, 3) if ms is not None else None
                else:
                    ms = tokens = None
                event = self._event(
                    job,
                    "step",
                    m.StepEventData(
                        addr=addr,
                        phase=phase,
                        cache_status=status,
                        ms=ms,
                        tokens=tokens,
                        sample=data["sample"],
                        branch=data["branch"],
                    ),
                )
                self._emit(job, event)
            elif raw["event"] == "outcome":
                event = self._event(
                    job,
                    "outcome",
                    m.OutcomeEventData(
                        run_id=data["run_id"],
                        sample=data["sample"],
                        branch=data["branch"],
                        passed=bool(data["passed"]),
                        reason=data.get("reason"),
                        is_control=data["branch"] == "control",
                    ),
                )
                self._emit(job, event)

        error_body: m.ErrorBody | None = None
        async with self.semaphore:
            job.status = "running"
            self._set_status(agent, job.fork_id, "running")
            try:
                adapter = agent.adapter()
                agent_fn = adapter.factory(base_id)
                edits = [Edit(e.addr, e.kind, e.value, e.known_good) for e in request.edits]
                batch = await ReplayEngine(agent.recorder).replay(
                    base_id,
                    agent_fn,
                    edits=edits,
                    mode=request.mode,
                    samples=request.samples,
                    control=request.control,
                    branch_name=job.branch_name,
                    event_callback=on_event,
                    fork_id=job.fork_id,
                    resample_from=request.resample_from,
                )
            except Exception as error:  # every failure becomes a terminal error event
                logger.exception("fork %s failed", job.fork_id)
                error_body = classify_replay_error(error).body()
                batch = None
        if batch is not None:
            savings = self.savings(job.agent, job.fork_id, base_id, request)
            summary = m.SummaryEventData(
                fork_id=job.fork_id,
                reexecuted=savings.reexecuted,
                cached=savings.cached,
                invalidated=savings.invalidated,
                total_steps=savings.total_steps,
                calls_reexecuted=savings.calls_reexecuted,
                calls_cached=savings.calls_cached,
                tokens_saved=batch.tokens_saved,
                ms_saved=round(batch.ms_saved, 3),
                fix_rate=batch.fix_pass_rate,
                fix_ci=batch.fix_interval,
                control_rate=batch.control_pass_rate,
                control_ci=batch.control_interval,
                k=request.samples,
                verdict=batch.verdict,
                preview=not batch.controls,
            )
            self._emit(job, self._event(job, "summary", summary))
            job.status = "complete"
        else:
            self._emit(job, self._event(job, "error", error_body))
            job.status = "error"
        problems = m.check_event_stream(job.events)
        if problems and job.status == "complete":
            logger.warning("fork %s stream problems: %s", job.fork_id, problems)
        agent.database.execute(
            """
            UPDATE fork_meta SET status = ?, events_json = ?, error_json = ?, completed_at = ?
            WHERE fork_id = ?
            """,
            (
                job.status,
                json.dumps([event.model_dump(mode="json") for event in job.events]),
                error_body.model_dump_json() if error_body else None,
                _now().isoformat(),
                job.fork_id,
            ),
        )
        service.ensure_index()
        with service.lock:
            service.cache.rows = None
        job.done.set()
        for queue in job.subscribers:
            queue.put_nowait(None)

    def _set_status(self, agent: Any, fork_id: str, status: str) -> None:
        agent.database.execute(
            "UPDATE fork_meta SET status = ? WHERE fork_id = ?", (status, fork_id)
        )

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------

    def _meta(self, fork_id: str) -> tuple[Any, dict[str, Any]]:
        for agent in self.service.agents.values():
            row = agent.database.one("SELECT * FROM fork_meta WHERE fork_id = ?", (fork_id,))
            if row is not None:
                return agent, row
        raise not_found("fork", fork_id)

    async def stream(self, fork_id: str) -> AsyncIterator[Any]:
        job = self.jobs.get(fork_id)
        if job is None:
            _, meta = self._meta(fork_id)
            if not meta["events_json"]:
                raise ApiError("conflict", "This fork was interrupted before it finished.")
            for event in json.loads(meta["events_json"]):
                yield event
            return
        queue: asyncio.Queue = asyncio.Queue()
        sent = list(job.events)
        if not job.done.is_set():
            job.subscribers.append(queue)
        try:
            for event in sent:
                yield event.model_dump(mode="json")
            if job.done.is_set():
                for event in job.events[len(sent) :]:
                    yield event.model_dump(mode="json")
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                if event.id <= len(sent):
                    continue
                yield event.model_dump(mode="json")
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    async def wait(self, fork_id: str) -> None:
        job = self.jobs.get(fork_id)
        if job is not None:
            await job.done.wait()

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def savings(
        self, agent_name: str, fork_id: str, base_id: str, request: m.ForkRequest
    ) -> m.Savings:
        agent = self.service.agents[agent_name]
        edited = {edit.addr for edit in request.edits}
        roots = edited or ({request.resample_from} if request.resample_from else set())
        invalidated = descendants(agent.reader.edges(base_id), roots)
        run_id = f"{fork_id}-fix-0"
        steps = agent.reader.step_details(run_id)
        statuses = normalise_statuses(steps, invalidated, edited)
        live = {"live", "edited"}
        has_call = {step.addr: step.has_call for step in steps}
        return m.Savings(
            reexecuted=sum(status in live for status in statuses.values()),
            cached=sum(status == "cached" for status in statuses.values()),
            invalidated=len(invalidated),
            total_steps=len(steps),
            calls_reexecuted=sum(has_call[a] and s in live for a, s in statuses.items()),
            calls_cached=sum(has_call[a] and s == "cached" for a, s in statuses.items()),
            tokens_saved=0,
            ms_saved=0.0,
        )

    def _fidelity(
        self, agent: Any, fork_id: str, base_id: str, request: m.ForkRequest
    ) -> float | None:
        base = {
            row["addr"]: row
            for row in agent.database.query(
                "SELECT addr, seq, state_before FROM steps WHERE run_id = ?", (base_id,)
            )
        }
        roots = {e.addr for e in request.edits} or {request.resample_from}
        first = min((base[a]["seq"] for a in roots if a in base), default=None)
        if first is None:
            return None
        replayed = agent.database.query(
            "SELECT addr, seq, state_before FROM steps WHERE run_id = ? AND seq < ?",
            (f"{fork_id}-fix-0", first),
        )
        if not replayed:
            return 1.0
        same = sum(
            base.get(row["addr"], {}).get("state_before") == row["state_before"] for row in replayed
        )
        return same / len(replayed)

    def summary(self, fork_id: str) -> m.ForkSummary:
        agent, meta = self._meta(fork_id)
        request = m.ForkRequest.model_validate_json(meta["request_json"])
        fork = agent.database.one("SELECT * FROM forks WHERE fork_id = ?", (fork_id,))
        runs = agent.database.query(
            "SELECT run_id, outcome, score, checker_reason FROM runs WHERE fork_id = ? ORDER BY run_id",
            (fork_id,),
        )

        def refs(branch: m.Branch) -> list[m.ForkRunRef]:
            prefix = f"{fork_id}-{branch}-"
            chosen = sorted(
                (r for r in runs if r["run_id"].startswith(prefix)),
                key=lambda r: int(r["run_id"].rsplit("-", 1)[1]),
            )
            return [
                m.ForkRunRef(
                    run_id=r["run_id"],
                    sample=int(r["run_id"].rsplit("-", 1)[1]),
                    branch=branch,
                    passed=r["outcome"] == "passed",
                    score=r["score"],
                    reason=r["checker_reason"],
                )
                for r in chosen
                if r["outcome"] is not None
            ]

        edited_runs, control_runs = refs("fix"), refs("control")
        status: m.ForkStatus = meta["status"]
        job = self.jobs.get(fork_id)
        if job is not None:
            status = job.status
        complete = status == "complete" and fork is not None
        fix = control = None
        verdict = None
        savings = None
        fidelity = None
        if complete:
            fix = m.pass_rate(sum(r.passed for r in edited_runs), len(edited_runs))
            if control_runs:
                control = m.pass_rate(sum(r.passed for r in control_runs), len(control_runs))
            verdict = fork["verdict"]
            savings = self.savings(agent.name, fork_id, meta["base_run_id"], request).model_copy(
                update={
                    "tokens_saved": fork["tokens_saved"],
                    "ms_saved": round(fork["ms_saved"], 3),
                }
            )
            fidelity = self._fidelity(agent, fork_id, meta["base_run_id"], request)
        error = m.ErrorBody.model_validate_json(meta["error_json"]) if meta["error_json"] else None
        return m.ForkSummary(
            fork_id=fork_id,
            base_run_id=meta["base_run_id"],
            branch_name=fork["branch_name"] if fork else (job.branch_name if job else fork_id[:8]),
            status=status,
            hypothesis=meta["hypothesis"],
            edits=request.edits,
            mode=request.mode,
            samples=request.samples,
            control=request.control,
            created_at=datetime.fromisoformat(meta["created_at"]),
            completed_at=datetime.fromisoformat(meta["completed_at"])
            if meta["completed_at"]
            else None,
            savings=savings,
            fix=fix,
            control_result=control,
            verdict=verdict,
            preview=complete and control is None,
            replay_fidelity=fidelity,
            edited_runs=edited_runs,
            control_runs=control_runs,
            export_available=verdict == "VERIFIED",
            error=error,
        )

    def forks_of(self, run_id: str) -> list[m.ForkSummary]:
        agent = self.service.agent_of(run_id)
        rows = agent.database.query(
            "SELECT fork_id FROM fork_meta WHERE base_run_id = ? ORDER BY created_at", (run_id,)
        )
        return [self.summary(row["fork_id"]) for row in rows]

    def timeline(self, run_id: str) -> m.ForkTimeline:
        base = self.service.run_detail(run_id).run
        entries = [
            m.TimelineEntry(
                kind="original",
                label="Original run",
                run_id=run_id,
                fork_id=None,
                parent_fork_id=None,
                outcome=base.status,
                score=base.score,
                checker_reason=base.checker_reason,
                pass_rate=None,
                verdict=None,
                preview=False,
                savings=None,
                edit_summary=None,
            )
        ]
        forks = self.forks_of(run_id)
        for fork in forks:
            first_fix = fork.edited_runs[0] if fork.edited_runs else None
            if fork.control and fork.control_runs:
                first = fork.control_runs[0]
                entries.append(
                    m.TimelineEntry(
                        kind="control",
                        label=f"Unchanged control ×{fork.samples}",
                        run_id=first.run_id,
                        fork_id=None,
                        parent_fork_id=fork.fork_id,
                        outcome="passed" if first.passed else "failed",
                        score=first.score,
                        checker_reason=first.reason,
                        pass_rate=fork.control_result,
                        verdict=None,
                        preview=False,
                        savings=None,
                        edit_summary=None,
                    )
                )
            entries.append(
                m.TimelineEntry(
                    kind="fork",
                    label=fork.branch_name,
                    run_id=first_fix.run_id if first_fix else None,
                    fork_id=fork.fork_id,
                    parent_fork_id=None,
                    outcome=("passed" if first_fix.passed else "failed") if first_fix else None,
                    score=first_fix.score if first_fix else None,
                    checker_reason=first_fix.reason if first_fix else None,
                    pass_rate=fork.fix,
                    verdict=fork.verdict,
                    preview=fork.preview,
                    savings=fork.savings,
                    edit_summary=edit_summary(fork.edits)
                    or f"re-run unchanged from {fork_request_resample(self, fork.fork_id)}",
                )
            )
        return m.ForkTimeline(base_run_id=run_id, entries=entries, forks=forks)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verification_for(self, run_id: str) -> m.Verification | None:
        try:
            forks = [f for f in self.forks_of(run_id) if f.edits]
        except ApiError:
            return None
        if not forks:
            return None
        complete = [f for f in forks if f.status == "complete"]
        verified = [f for f in complete if f.verdict == "VERIFIED"]
        fork = (verified or complete or forks)[-1]
        if fork.status in {"queued", "running"}:
            return m.Verification(
                verdict=None,
                status="running",
                fork_id=fork.fork_id,
                edit_addr=fork.edits[0].addr,
                k=fork.samples,
                fix=None,
                control=None,
                replay_fidelity=None,
                preview=not fork.control,
                explanation="Replaying the edit and its paired control.",
            )
        if fork.status != "complete" or fork.fix is None:
            return None
        return m.Verification(
            verdict=fork.verdict,
            status="complete",
            fork_id=fork.fork_id,
            edit_addr=fork.edits[0].addr,
            k=fork.samples,
            fix=fork.fix,
            control=fork.control_result,
            replay_fidelity=fork.replay_fidelity,
            preview=fork.preview,
            explanation=explain_verdict(fork),
        )

    async def start_verify(self, run_id: str, request: m.VerifyRequest) -> m.VerifyJob:
        diagnosis = self.service.diagnosis(run_id)
        fixes = {fix.addr: fix for fix in diagnosis.proposed_fixes}
        if request.suspects:
            addrs = list(request.suspects)
        else:
            addrs = [fix.addr for fix in diagnosis.proposed_fixes][:3]
        if not addrs:
            raise ApiError(
                "not_applicable",
                "There is no proposed fix to verify for this run.",
                hint="Open Fork and fix and write an edit by hand.",
            )
        candidates = []
        for addr in addrs:
            fix = fixes.get(addr)
            candidates.append(
                m.VerifyCandidate(
                    addr=addr,
                    status="queued" if fix else "failed",
                    fork_id=None,
                    edit=fix.edit if fix else None,
                    edit_source=fix.source if fix else None,
                    fix=None,
                    control_result=None,
                    verdict=None,
                    note=None if fix else "No rule-derived, oracle or twin value for this step.",
                )
            )
        job = m.VerifyJob(
            job_id=uuid.uuid4().hex,
            run_id=run_id,
            status="queued",
            samples=request.samples,
            candidates=candidates,
            best_addr=None,
            created_at=_now(),
            completed_at=None,
            error=None,
        )
        state = VerifyState(job)
        self.verify_jobs[job.job_id] = state
        state.task = asyncio.create_task(self._verify(state))
        return job

    async def _verify(self, state: VerifyState) -> None:
        job = state.job
        state.job = job = job.model_copy(update={"status": "running"})
        updated = list(job.candidates)
        try:
            for index, candidate in enumerate(updated):
                if candidate.edit is None:
                    continue
                updated[index] = candidate = candidate.model_copy(update={"status": "running"})
                state.job = job.model_copy(update={"candidates": list(updated)})
                try:
                    created = await self.create(
                        m.ForkRequest(
                            base_run_id=job.run_id,
                            edits=[candidate.edit],
                            mode="cone",
                            samples=job.samples,
                            control=True,
                            hypothesis=f"Verify {candidate.addr} ({candidate.edit_source})",
                        )
                    )
                    await self.wait(created.fork_id)
                    fork = self.summary(created.fork_id)
                    updated[index] = candidate.model_copy(
                        update={
                            "status": "complete" if fork.status == "complete" else "failed",
                            "fork_id": fork.fork_id,
                            "fix": fork.fix,
                            "control_result": fork.control_result,
                            "verdict": fork.verdict,
                            "note": explain_verdict(fork)
                            if fork.fix
                            else (fork.error.message if fork.error else None),
                        }
                    )
                except ApiError as error:
                    updated[index] = candidate.model_copy(
                        update={"status": "failed", "note": error.message}
                    )
                state.job = job.model_copy(update={"candidates": list(updated)})
            verified = [
                c for c in updated if c.verdict == "VERIFIED" and c.fix and c.control_result
            ]
            best = max(verified, key=lambda c: c.fix.rate - c.control_result.rate, default=None)
            state.job = job.model_copy(
                update={
                    "status": "complete",
                    "candidates": updated,
                    "best_addr": best.addr if best else None,
                    "completed_at": _now(),
                }
            )
        except Exception as error:
            logger.exception("verify job %s failed", job.job_id)
            state.job = job.model_copy(
                update={
                    "status": "failed",
                    "candidates": updated,
                    "completed_at": _now(),
                    "error": ApiError("internal_error", str(error)).body(),
                }
            )

    def get_job(self, job_id: str) -> m.VerifyJob:
        state = self.verify_jobs.get(job_id)
        if state is None:
            raise not_found("job", job_id)
        return state.job


def fork_request_resample(manager: ForkManager, fork_id: str) -> str:
    _, meta = manager._meta(fork_id)
    request = m.ForkRequest.model_validate_json(meta["request_json"])
    return request.resample_from or "?"


def explain_verdict(fork: m.ForkSummary) -> str:
    if fork.fix is None:
        return "The replay did not finish."
    fix = f"{fork.fix.passed}/{fork.fix.total}"
    if fork.control_result is None:
        return f"Passed {fix} with the edit. No paired control ran, so this is a preview, not a verdict."
    control = f"{fork.control_result.passed}/{fork.control_result.total}"
    if fork.verdict == "VERIFIED":
        return (
            f"With the edit the run passed {fix}; unchanged it passed {control}. The 95% "
            "intervals do not overlap, so this step is a cause of the failure."
        )
    if fork.verdict == "REFUTED":
        return (
            f"A known-good value replaced this step's output, yet the run passed only {fix} "
            f"(control {control}). This step is not what made the run fail."
        )
    return (
        f"Edit {fix} vs control {control}: the 95% intervals overlap. Add samples "
        "(K=10) or try a better edit to separate them."
    )
