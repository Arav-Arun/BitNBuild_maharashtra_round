import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const sourceUrl = new URL("../lib/api.ts", import.meta.url);
const compiled = ts.transpileModule(readFileSync(sourceUrl, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
}).outputText;

// Execute the actual API client with controlled network responses, without a browser.
function client(fetch, apiUrl = "https://blackbox.invalid") {
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports, require: createRequire(sourceUrl),
    process: { env: { NODE_ENV: "production", NEXT_PUBLIC_API_URL: apiUrl } },
    fetch, Headers, URLSearchParams,
  }, { filename: "api.js" });
  return module.exports;
}

test("an outage cannot substitute another run's diagnosis or comparison", async () => {
  const { api } = client(async () => { throw new Error("offline"); });
  assert.equal((await api("/runs/tc-0001/diagnosis")).run_id, "tc-0001");
  await assert.rejects(api("/runs/real-run/diagnosis"), (error) => error.code === "unavailable");
  await assert.rejects(api("/runs/real-run/report"), (error) => error.code === "unavailable");
  await assert.rejects(api("/diff?a=real-run&b=real-fix"), (error) => error.code === "unavailable");
});

test("the static showcase serves only its exact run and comparison", async () => {
  const { api } = client(() => { throw new Error("a static build must not fetch"); }, "");
  assert.equal((await api("/runs/tc-0001")).run.run_id, "tc-0001");
  assert.equal((await api("/diff?a=tc-0001&b=fk-0001-fix-0")).right_run_id, "fk-0001-fix-0");
  await assert.rejects(api("/runs/tc-9999"), (error) => error.code === "unavailable");
});

test("cancelled requests do not turn into fallback data", async () => {
  const controller = new AbortController();
  const abort = new Error("cancelled");
  const { api } = client(async () => { controller.abort(); throw abort; });
  await assert.rejects(api("/runs", { signal: controller.signal }), (error) => error === abort);
});

test("Headers objects and JSON content type reach the API", async () => {
  const { api } = client(async (_url, init) => {
    assert.equal(init.headers.get("Authorization"), "Bearer test");
    assert.equal(init.headers.get("Content-Type"), "application/json");
    return Response.json({ ok: true });
  });
  assert.equal((await api("/tasks/run", {
    method: "POST", body: "{}", headers: new Headers({ Authorization: "Bearer test" }),
  })).ok, true);
});

test("invalid successful responses produce an actionable error", async () => {
  const { api } = client(async () => new Response("<html>not an API</html>"));
  await assert.rejects(api("/health"), (error) => error.code === "invalid_response");
});

test("server error hints are retained for display", async () => {
  const { api, errorText } = client(async () => Response.json({
    error: { code: "bad_request", message: "Missing dates.", hint: "Use YYYY-MM-DD." },
  }, { status: 400 }));
  await assert.rejects(api("/tasks/run", { method: "POST", body: "{}" }), (error) => {
    assert.equal(errorText(error), "Missing dates. Use YYYY-MM-DD.");
    return error.code === "bad_request";
  });
});
