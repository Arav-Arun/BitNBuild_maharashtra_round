"""Immutable replay branches with exact-hash caching and causal cone invalidation."""

from __future__ import annotations

import json
import math
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from blackbox.sdk import Recorder, RunSession
from blackbox.sdk.runtime import chat_request_key

ReplayMode = Literal["cone", "prefix", "full"]
Verdict = Literal["VERIFIED", "REFUTED", "INCONCLUSIVE"]


class ReplayDivergence(RuntimeError):
    def __init__(self, addr: str, detail: str = "state does not match the recording") -> None:
        super().__init__(f"Replay diverged at {addr}: {detail}")
        self.addr = addr


@dataclass(frozen=True, slots=True)
class Edit:
    addr: str
    kind: Literal[
        "override_output",
        "patch_tool_args",
        "patch_prompt",
        "swap_model",
        "patch_tool_result",
        "ghost_hint",
        "rerun",
    ]
    value: Any
    known_good: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "addr": self.addr,
            "kind": self.kind,
            # A ghost hint must never be persisted, or the trace would carry the answer.
            "value": "<hidden>" if self.kind == "ghost_hint" else self.value,
            "known_good": self.known_good,
        }


def override_output(addr: str, value: Any, *, known_good: bool = False) -> Edit:
    return Edit(addr, "override_output", value, known_good)


def patch_tool_args(addr: str, args: Mapping[str, Any], *, known_good: bool = False) -> Edit:
    return Edit(addr, "patch_tool_args", dict(args), known_good)


def patch_prompt(addr: str, patch: Any, *, known_good: bool = False) -> Edit:
    return Edit(addr, "patch_prompt", patch, known_good)


def swap_model(addr: str, model: str, *, known_good: bool = False) -> Edit:
    return Edit(addr, "swap_model", model, known_good)


def patch_tool_result(addr: str, value: Any, *, known_good: bool = False) -> Edit:
    return Edit(addr, "patch_tool_result", value, known_good)


def ghost_hint(addr: str, hint: str) -> Edit:
    """Re-run an LLM step live with a hidden instruction that is never recorded.

    The stored request, request key and cassette stay those of the clean prompt;
    only the live provider call sees the hint, so the model writes the fault in
    its own style without leaving the instruction in the trace.
    """
    return Edit(addr, "ghost_hint", hint)


def rerun(addr: str) -> Edit:
    """Execute the step live again with its unchanged request (a retry)."""
    return Edit(addr, "rerun", None)


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one sample")
    proportion = successes / total
    denominator = 1 + (z * z) / total
    center = (proportion + (z * z) / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) / total) + (z * z) / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(slots=True)
class ReplayRun:
    run_id: str
    outcome: str | None
    score: float | None
    reason: str | None
    statuses: dict[str, str]
    state_after: str | None


@dataclass(slots=True)
class ReplayBatch:
    fork_id: str
    base_run_id: str
    mode: ReplayMode
    edited: list[ReplayRun]
    controls: list[ReplayRun]
    invalidated: set[str]
    reexecuted_steps: int
    cached_steps: int
    tokens_saved: int
    ms_saved: float
    fix_pass_rate: float
    fix_interval: tuple[float, float]
    control_pass_rate: float | None
    control_interval: tuple[float, float] | None
    verdict: Verdict | None


def _apply_prompt_patch(request: dict[str, Any], patch: Any) -> dict[str, Any]:
    updated = dict(request)
    if isinstance(patch, str):
        messages = list(updated.get("messages", []))
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": f"{messages[0].get('content', '')}\n{patch}".strip(),
            }
        else:
            messages.insert(0, {"role": "system", "content": patch})
        updated["messages"] = messages
    elif isinstance(patch, Mapping):
        updated.update(dict(patch))
    else:
        raise TypeError("prompt patch must be a string or mapping")
    return updated


@dataclass(slots=True)
class _ReplayPolicy:
    recorder: Recorder
    base_steps: dict[str, dict[str, Any]]
    edits: dict[str, Edit]
    mode: ReplayMode
    invalidated: set[str]
    force_live: set[str] = field(default_factory=set)
    event_callback: Callable[[dict[str, Any]], None] | None = None
    statuses: dict[str, str] = field(default_factory=dict)
    last_state_after: str | None = None
    base_seed: int | None = None
    sample_seed: int | None = None
    first_edit_seq: float = field(init=False)

    def __post_init__(self) -> None:
        edited_sequences = [
            self.base_steps[addr]["seq"] for addr in self.edits if addr in self.base_steps
        ]
        if not edited_sequences:
            edited_sequences = [
                self.base_steps[addr]["seq"] for addr in self.force_live if addr in self.base_steps
            ]
        self.first_edit_seq = min(edited_sequences, default=math.inf)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def before_step(self, addr: str, seq: int, state_before: str) -> None:
        base = self.base_steps.get(addr)
        if base is None:
            if seq < self.first_edit_seq:
                raise ReplayDivergence(addr, "step is absent from the base run")
            return
        if seq < self.first_edit_seq and state_before != base["state_before"]:
            raise ReplayDivergence(addr)
        if addr in self.invalidated or addr in self.force_live:
            self._emit(
                {
                    "event": "step",
                    "data": {
                        "addr": addr,
                        "phase": "queued",
                        "cache_status": "invalidated",
                    },
                }
            )

    def after_step(self, addr: str, cache_status: str, state_after: str) -> None:
        self.statuses[addr] = cache_status
        self.last_state_after = state_after
        self._emit(
            {
                "event": "step",
                "data": {
                    "addr": addr,
                    "phase": "done",
                    "cache_status": cache_status,
                },
            }
        )

    def transform_chat(self, addr: str, request: dict[str, Any]) -> dict[str, Any]:
        edit = self.edits.get(addr)
        if edit is not None:
            if edit.kind == "patch_prompt":
                request = _apply_prompt_patch(request, edit.value)
            elif edit.kind == "swap_model":
                request = {**request, "model": str(edit.value)}

        # Sampling must not change otherwise identical cached prefix or sibling
        # requests. Only normalize the automatic run.seed, and only when doing
        # so recovers the exact base request key. Changed/live requests retain
        # their sample seed; explicit custom seeds and other drift stay visible.
        base = self.base_steps.get(addr)
        explicit_seed_edit = (
            edit is not None
            and edit.kind == "patch_prompt"
            and isinstance(edit.value, Mapping)
            and "seed" in edit.value
        )
        forced = (
            self.mode == "full"
            or addr in self.force_live
            or (self.mode == "prefix" and base is not None and base["seq"] >= self.first_edit_seq)
        )
        if (
            not forced
            and not explicit_seed_edit
            and base is not None
            and self.base_seed is not None
            and self.sample_seed is not None
            and request.get("seed") == self.sample_seed
        ):
            original_seed_request = {**request, "seed": self.base_seed}
            if chat_request_key(original_seed_request, addr) == base["request_key"]:
                return original_seed_request
        return request

    def transform_tool(self, addr: str, request: dict[str, Any]) -> dict[str, Any]:
        edit = self.edits.get(addr)
        if edit is None or edit.kind != "patch_tool_args":
            return request
        if not isinstance(edit.value, Mapping):
            raise TypeError("tool argument patch must be a mapping")
        return {**request, "args": {**request["args"], **dict(edit.value)}}

    def live_chat_request(self, addr: str, request: dict[str, Any]) -> dict[str, Any]:
        """The request actually sent to the provider; differs only for ghost hints."""
        edit = self.edits.get(addr)
        if edit is None or edit.kind != "ghost_hint":
            return request
        return _apply_prompt_patch(request, str(edit.value))

    def _load_recorded(self, addr: str) -> Any:
        step = self.base_steps.get(addr)
        if step is None or not step.get("output_hash"):
            raise ReplayDivergence(addr, "recorded output is unavailable")
        try:
            return self.recorder.store.load_json(step["output_hash"])
        except (OSError, ValueError) as error:
            raise ReplayDivergence(addr, str(error)) from error

    def _load_cassette(self, addr: str, request_key: str) -> tuple[bool, Any]:
        cassette = self.recorder.database.one(
            "SELECT response_hash FROM cassette WHERE request_key = ?", (request_key,)
        )
        if cassette is None:
            return False, None
        try:
            return True, self.recorder.store.load_json(cassette["response_hash"])
        except (OSError, ValueError) as error:
            raise ReplayDivergence(addr, str(error)) from error

    async def resolve(
        self,
        addr: str,
        kind: str,
        request_key: str,
        live: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, str]:
        del kind
        edit = self.edits.get(addr)
        if edit is not None and edit.kind in {"override_output", "patch_tool_result"}:
            return edit.value, "edited"
        if edit is not None and edit.kind in {"ghost_hint", "rerun"}:
            return await live(), "edited"

        base = self.base_steps.get(addr)
        if addr in self.force_live:
            return await live(), "live"
        if self.mode == "full":
            return await live(), "live"
        if self.mode == "prefix" and base is not None and base["seq"] >= self.first_edit_seq:
            return await live(), "live"
        if base is None:
            # A step the base run never reached (it failed earlier) can still be an
            # exact-hash cassette hit, e.g. one recorded by a verified fix of this run.
            cassette_hit, cassette_output = self._load_cassette(addr, request_key)
            if cassette_hit:
                return cassette_output, "cached"
            return await live(), "live"
        if base["request_key"] == request_key:
            return self._load_recorded(addr), "cached"
        cassette_hit, cassette_output = self._load_cassette(addr, request_key)
        if cassette_hit:
            return cassette_output, "cached"
        if base["request_key"] is None and base.get("error_type"):
            # The recorded call raised before completing, so there is no request to
            # compare and no response to serve: re-running it is the only faithful
            # reproduction of that crash.
            return await live(), "live"
        if base["seq"] < self.first_edit_seq:
            raise ReplayDivergence(addr, "request hash changed before the edit")
        return await live(), "live"


class ReplayEngine:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def _base(self, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        run = self.recorder.database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if run is None:
            raise KeyError(f"unknown base run: {run_id}")
        steps = self.recorder.database.query(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        if not steps:
            raise ValueError(f"base run has no recorded steps: {run_id}")
        return run, steps

    def _descendants(self, run_id: str, roots: set[str]) -> set[str]:
        edges = self.recorder.database.query(
            "SELECT src_addr, dst_addr FROM edges WHERE run_id = ?", (run_id,)
        )
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge["src_addr"], set()).add(edge["dst_addr"])
        seen = set(roots)
        queue = deque(roots)
        while queue:
            source = queue.popleft()
            for destination in adjacency.get(source, set()):
                if destination not in seen:
                    seen.add(destination)
                    queue.append(destination)
        return seen

    async def _one(
        self,
        *,
        base: dict[str, Any],
        base_steps: dict[str, dict[str, Any]],
        agent_fn: Callable[[RunSession], Awaitable[Any] | Any],
        policy: _ReplayPolicy,
        fork_id: str,
        sample_index: int,
        branch: str,
    ) -> ReplayRun:
        run_id = f"{fork_id}-{branch}-{sample_index}"
        seed = int(base.get("seed") or 0) + sample_index
        policy.base_seed = int(base.get("seed") or 0)
        policy.sample_seed = seed
        with self.recorder.run(
            base["agent"],
            base["task_id"],
            seed,
            run_id=run_id,
            model=base.get("model"),
            parent_run_id=base["run_id"],
            fork_id=fork_id,
            replay_policy=policy,
        ) as run:
            result = agent_fn(run)
            if hasattr(result, "__await__"):
                await result
        row = self.recorder.database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise AssertionError("replay run was not persisted")
        policy._emit(
            {
                "event": "outcome",
                "data": {
                    "run_id": run_id,
                    "sample": sample_index,
                    "branch": branch,
                    "passed": row["outcome"] == "passed",
                    "reason": row["checker_reason"],
                },
            }
        )
        return ReplayRun(
            run_id=run_id,
            outcome=row["outcome"],
            score=row["score"],
            reason=row["checker_reason"],
            statuses=dict(policy.statuses),
            state_after=policy.last_state_after,
        )

    async def replay(
        self,
        base_run_id: str,
        agent_fn: Callable[[RunSession], Awaitable[Any] | Any],
        *,
        edits: Sequence[Edit] = (),
        mode: ReplayMode = "cone",
        samples: int = 1,
        control: bool = False,
        control_scope: Literal["cone", "downstream"] = "cone",
        branch_name: str | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ReplayBatch:
        """Replay ``base_run_id`` with ``edits``, optionally beside no-edit controls.

        A control re-runs the invalidated cone live without the edit. With
        ``control_scope="cone"`` the edited steps themselves re-run too (the forge's
        check that a passing run reproduces); with ``"downstream"`` they keep their
        recorded output, so the control reproduces the run *as recorded*, which is
        what a verifier compares a fix against.
        """
        if mode not in {"cone", "prefix", "full"}:
            raise ValueError("mode must be cone, prefix, or full")
        if control_scope not in {"cone", "downstream"}:
            raise ValueError("control_scope must be cone or downstream")
        if samples < 1:
            raise ValueError("samples must be positive")
        addresses = [edit.addr for edit in edits]
        if len(addresses) != len(set(addresses)):
            raise ValueError("only one edit per step is supported")
        base, ordered_steps = self._base(base_run_id)
        base_steps = {step["addr"]: step for step in ordered_steps}
        unknown = set(addresses) - set(base_steps)
        if unknown:
            raise ValueError(f"edit addresses are absent from the base run: {sorted(unknown)}")
        for edit in edits:
            if edit.kind == "ghost_hint" and base_steps[edit.addr]["kind"] != "llm":
                raise ValueError(f"ghost hints only apply to LLM steps: {edit.addr}")
        edit_map = {edit.addr: edit for edit in edits}
        invalidated = self._descendants(base_run_id, set(addresses))
        control_live = (
            invalidated - set(addresses) if control_scope == "downstream" else set(invalidated)
        )
        fork_id = uuid.uuid4().hex
        edited_runs: list[ReplayRun] = []
        control_runs: list[ReplayRun] = []
        policies: list[_ReplayPolicy] = []

        for index in range(samples):
            policy = _ReplayPolicy(
                self.recorder,
                base_steps,
                edit_map,
                mode,
                invalidated,
                event_callback=event_callback,
            )
            policies.append(policy)
            edited_runs.append(
                await self._one(
                    base=base,
                    base_steps=base_steps,
                    agent_fn=agent_fn,
                    policy=policy,
                    fork_id=fork_id,
                    sample_index=index,
                    branch="fix",
                )
            )
            if control:
                control_policy = _ReplayPolicy(
                    self.recorder,
                    base_steps,
                    {},
                    "cone",
                    invalidated,
                    force_live=set(control_live),
                    event_callback=event_callback,
                )
                policies.append(control_policy)
                control_runs.append(
                    await self._one(
                        base=base,
                        base_steps=base_steps,
                        agent_fn=agent_fn,
                        policy=control_policy,
                        fork_id=fork_id,
                        sample_index=index,
                        branch="control",
                    )
                )

        fix_successes = sum(run.outcome == "passed" for run in edited_runs)
        fix_rate = fix_successes / samples
        fix_interval = wilson_interval(fix_successes, samples)
        control_rate: float | None = None
        control_interval: tuple[float, float] | None = None
        verdict: Verdict | None = None
        if control_runs:
            control_successes = sum(run.outcome == "passed" for run in control_runs)
            control_rate = control_successes / samples
            control_interval = wilson_interval(control_successes, samples)
            if samples >= 3 and fix_interval[0] > control_interval[1]:
                verdict = "VERIFIED"
            elif (
                samples >= 3
                and any(edit.known_good for edit in edits)
                and fix_successes == 0
                and fix_rate <= control_rate
            ):
                verdict = "REFUTED"
            else:
                verdict = "INCONCLUSIVE"

        reexecuted = sum(
            status in {"live", "edited"}
            for policy in policies
            for status in policy.statuses.values()
        )
        cached = sum(
            status == "cached" for policy in policies for status in policy.statuses.values()
        )
        tokens_saved = 0
        ms_saved = 0.0
        for policy in policies:
            for addr, status in policy.statuses.items():
                if status != "cached" or addr not in base_steps:
                    continue
                step = base_steps[addr]
                tokens_saved += int(step.get("tokens_in") or 0) + int(step.get("tokens_out") or 0)
                ms_saved += float(step.get("latency_ms") or 0)

        self.recorder.database.execute(
            """
            INSERT INTO forks(
                fork_id, base_run_id, branch_name, edits_json, mode, samples,
                reexec_steps, cached_steps, invalidated_steps, tokens_saved, ms_saved,
                fix_pass_rate, fix_ci_low, fix_ci_high, control_pass_rate,
                control_ci_low, control_ci_high, verdict, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fork_id,
                base_run_id,
                branch_name or f"fork-{fork_id[:8]}",
                json.dumps([edit.as_dict() for edit in edits], sort_keys=True),
                mode,
                samples,
                reexecuted,
                cached,
                len(invalidated),
                tokens_saved,
                ms_saved,
                fix_rate,
                fix_interval[0],
                fix_interval[1],
                control_rate,
                control_interval[0] if control_interval else None,
                control_interval[1] if control_interval else None,
                verdict,
                datetime.now(UTC).isoformat(),
            ),
        )
        summary = {
            "event": "summary",
            "data": {
                "fork_id": fork_id,
                "reexecuted": reexecuted,
                "cached": cached,
                "tokens_saved": tokens_saved,
                "ms_saved": ms_saved,
                "fix_rate": fix_rate,
                "fix_ci": fix_interval,
                "control_rate": control_rate,
                "control_ci": control_interval,
                "verdict": verdict,
            },
        }
        if event_callback is not None:
            event_callback(summary)
        return ReplayBatch(
            fork_id=fork_id,
            base_run_id=base_run_id,
            mode=mode,
            edited=edited_runs,
            controls=control_runs,
            invalidated=invalidated,
            reexecuted_steps=reexecuted,
            cached_steps=cached,
            tokens_saved=tokens_saved,
            ms_saved=ms_saved,
            fix_pass_rate=fix_rate,
            fix_interval=fix_interval,
            control_pass_rate=control_rate,
            control_interval=control_interval,
            verdict=verdict,
        )


async def replay(
    recorder: Recorder,
    base_run_id: str,
    agent_fn: Callable[[RunSession], Awaitable[Any] | Any],
    **kwargs: Any,
) -> ReplayBatch:
    """Convenience wrapper around :class:`ReplayEngine`."""

    return await ReplayEngine(recorder).replay(base_run_id, agent_fn, **kwargs)


__all__ = [
    "Edit",
    "ReplayBatch",
    "ReplayRun",
    "ReplayDivergence",
    "ReplayEngine",
    "ghost_hint",
    "override_output",
    "patch_prompt",
    "patch_tool_args",
    "patch_tool_result",
    "replay",
    "rerun",
    "swap_model",
    "wilson_interval",
]
