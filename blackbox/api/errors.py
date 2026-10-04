"""One error type for the service; the HTTP and MCP layers turn it into an ``ErrorBody``."""

from __future__ import annotations

from typing import Any

from server import models as m

STATUS: dict[str, int] = {
    "bad_request": 400,
    "validation_error": 422,
    "not_found": 404,
    "conflict": 409,
    "live_call_refused": 409,
    "replay_divergence": 409,
    "not_verified": 409,
    "not_applicable": 409,
    "rate_limited": 429,
    "unavailable": 503,
    "unsupported": 501,
    "internal_error": 500,
}


class ApiError(Exception):
    def __init__(
        self,
        code: m.ErrorCode,
        message: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.context = context
        self.status = status or STATUS[code]

    def body(self, request_id: str | None = None) -> m.ErrorBody:
        return m.ErrorBody(
            code=self.code,
            message=self.message,
            status=self.status,
            hint=self.hint,
            issues=[],
            context=self.context,
            request_id=request_id,
        )


def not_found(what: str, ident: str, hint: str | None = None) -> ApiError:
    return ApiError("not_found", f"No {what} {ident!r}.", hint=hint, context={what: ident})


def classify_replay_error(error: BaseException) -> ApiError:
    """Map an exception raised while replaying to a stable error code."""
    text = str(error)
    if "recorded mode refuses live" in text:
        return ApiError(
            "live_call_refused",
            "This fork needs a live model call, and the server runs in RECORDED mode.",
            hint="Edit a step whose consumers are all cached, or run the API with MODE=offline.",
        )
    if "set GROQ_API_KEY" in text:
        return ApiError(
            "live_call_refused",
            "This run was recorded with a hosted model and no API key is configured.",
            hint="Set GROQ_API_KEY locally to replay it, or fork a run recorded offline.",
        )
    if type(error).__name__ == "ReplayDivergence":
        return ApiError(
            "replay_divergence",
            text,
            hint="The agent code or its inputs changed since recording; re-record the run.",
        )
    return ApiError("internal_error", f"Replay failed: {type(error).__name__}: {text}")
