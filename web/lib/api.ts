import agentsFixture from "../mocks/agents.json";
import diagnosisFixture from "../mocks/diagnosis.json";
import diffFixture from "../mocks/diff.json";
import evalFixture from "../mocks/eval.json";
import groupsFixture from "../mocks/failure-groups.json";
import healthFixture from "../mocks/health.json";
import labelFixture from "../mocks/label-queue.json";
import reportFixture from "../mocks/crash-report.json";
import runsFixture from "../mocks/runs.json";
import detailFixture from "../mocks/run-detail.json";

// In local development, the Next app and FastAPI run on separate ports. Keep
// the demo connected even when NEXT_PUBLIC_API_URL was not set in the shell.
const DEFAULT_DEV_API = process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "";
export const API = (process.env.NEXT_PUBLIC_API_URL || DEFAULT_DEV_API).replace(/\/$/, "");
/** What to do when the API cannot be reached: start it locally, or wait for a hosted one to wake. */
export const UNREACHABLE_HINT = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/.test(API)
  ? "Start it with make dev-api, then retry."
  : "A free-tier host can take about a minute to wake up; retry shortly.";
function recordedFallback(path: string): unknown {
  const [route] = path.split("?");
  if (route === "/health") return { ...healthFixture, static_bundle: true,
    capabilities: { ...healthFixture.capabilities, fork: false, verify: false, export_test: false, label: false },
    notes: [...healthFixture.notes, "Static recorded showcase is active; writes are disabled."] };
  if (route === "/agents") return agentsFixture;
  if (route === "/runs") return runsFixture;
  if (route === "/failure-groups") return groupsFixture;
  if (route === "/eval") return evalFixture;
  if (route === "/labels/queue") return labelFixture;
  if (route === "/diff") return diffFixture;
  if (route.endsWith("/report")) return reportFixture;
  if (route.endsWith("/diagnosis")) return diagnosisFixture;
  if (/^\/runs\/[^/]+$/.test(route) && route.includes("tc-")) return detailFixture;
  return undefined;
}

/** A failed API call with the contract's error envelope (code, hint) preserved. */
export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null,
    readonly hint: string | null,
  ) {
    super(message);
  }
}

/** "message. hint" for display; plain Error messages pass through. */
export function errorText(error: unknown): string {
  if (error instanceof ApiRequestError) return error.hint ? `${error.message} ${error.hint}` : error.message;
  return (error as Error)?.message || "Request failed.";
}

/** True when this build has no API to call: a static showcase deployment. */
export const STATIC_BUILD = API === "";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method || "GET";
  const fixture = () => (method === "GET" ? recordedFallback(path) : undefined);
  if (STATIC_BUILD) {
    // Without NEXT_PUBLIC_API_URL a production build would call itself and get HTML 404s.
    const data = fixture();
    if (data !== undefined) return data as T;
    throw new ApiRequestError("This static showcase has no API, so this view is unavailable.", 503, "unavailable",
      "Deploy with NEXT_PUBLIC_API_URL pointing at a running Black Box API.");
  }
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { ...init, cache: "no-store", headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers,
    } });
  } catch {
    const data = fixture();
    if (data !== undefined) return data as T;
    throw new ApiRequestError(`Cannot reach the Black Box API at ${API}.`, 0, "unavailable", UNREACHABLE_HINT);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiRequestError(
      body?.error?.message || `Request failed (${response.status})`,
      response.status,
      body?.error?.code ?? null,
      body?.error?.hint ?? null,
    );
  }
  return body as T;
}
