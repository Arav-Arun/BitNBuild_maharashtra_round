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
import hopragFixture from "../mocks/run-detail-hoprag.json";

// In local development, the Next app and FastAPI run on separate ports. Keep
// the demo connected even when NEXT_PUBLIC_API_URL was not set in the shell.
const DEFAULT_DEV_API = process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "";
export const API = (process.env.NEXT_PUBLIC_API_URL || DEFAULT_DEV_API).replace(/\/$/, "");
function recordedFallback(path: string): unknown {
  const [route, query = ""] = path.split("?");
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
  if (route.endsWith("/diagnosis")) return route.includes("hr-") ? undefined : diagnosisFixture;
  if (/^\/runs\/[^/]+$/.test(route)) {
    if (route.includes("hr-")) return hopragFixture;
    if (route.includes("tc-")) return detailFixture;
    const params = new URLSearchParams(query);
    if (params.has("limit") || params.has("q") || params.has("outcome")) return runsFixture;
  }
  return undefined;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method || "GET";
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { ...init, cache: "no-store", headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers,
    } });
  } catch (error) {
    const fixture = method === "GET" ? recordedFallback(path) : undefined;
    if (fixture !== undefined) return fixture as T;
    throw error;
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error?.message || `Request failed (${response.status})`);
  }
  return body as T;
}
