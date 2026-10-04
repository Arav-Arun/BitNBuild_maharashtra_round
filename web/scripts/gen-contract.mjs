// Regenerates lib/contract.ts from ../server/contract.schema.json.
//   node scripts/gen-contract.mjs           write lib/contract.ts
//   node scripts/gen-contract.mjs --check   exit 1 if lib/contract.ts is stale
// The schema is produced by `python -m server.contract_export`; do not edit the output by hand.
import { compile } from "json-schema-to-typescript";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const schemaPath = resolve(here, "../../server/contract.schema.json");
const outPath = resolve(here, "../lib/contract.ts");
const check = process.argv.includes("--check");

const bundle = JSON.parse(readFileSync(schemaPath, "utf8"));
const defs = bundle["$defs"];

const banner = [
  "/* eslint-disable */",
  "// GENERATED FILE. Source of truth: server/models.py.",
  "// Regenerate with `make contract` (python -m server.contract_export).",
].join("\n");

let body = await compile(
  { title: "ContractRoot", type: "object", additionalProperties: false, properties: {}, $defs: defs },
  "ContractRoot",
  {
    bannerComment: "",
    unreachableDefinitions: true,
    additionalProperties: false,
    format: true,
    style: { singleQuote: false, printWidth: 100, semi: true, trailingComma: "all" },
  },
);

// The empty root only exists to anchor the definitions, and the "referenced by" notes are noise.
const NOTE = /^ \* (This interface was referenced by `ContractRoot`'s JSON-Schema|via the `definition` ".*"\.)$/;
body = body
  .replace(/export interface ContractRoot \{\}\n?/, "")
  .split("\n")
  .filter((line) => !NOTE.test(line))
  .join("\n")
  .replace(/\/\*\*\n \*\/\n/g, "")
  .replace(/\n \*\n \*\//g, "\n */")
  .replace(/\n{3,}/g, "\n\n")
  .trim();

const lines = [banner, "", `export const CONTRACT_VERSION = ${JSON.stringify(bundle["x-contract-version"])};`, "", body, ""];

lines.push("// Named enumerations: runtime value lists plus the derived union types.");
for (const [name, values] of Object.entries(bundle["x-enums"])) {
  lines.push(`export const ${name}Values = ${JSON.stringify(values)} as const;`);
  lines.push(`export type ${name} = (typeof ${name}Values)[number];`);
}
lines.push("");
lines.push("// Named unions of generated interfaces.");
for (const [name, members] of Object.entries(bundle["x-unions"])) {
  lines.push(`export type ${name} = ${members.join(" | ")};`);
}
lines.push("");
lines.push("export interface EndpointSpec {");
lines.push("  method: string;");
lines.push("  path: string;");
lines.push("  summary: string;");
lines.push("  request: string | null;");
lines.push("  response: string;");
lines.push("  statuses: readonly number[];");
lines.push("  query: readonly string[];");
lines.push("  extension: boolean;");
lines.push("}");
lines.push("");
lines.push("export const ENDPOINTS: readonly EndpointSpec[] = [");
for (const endpoint of bundle["x-endpoints"]) {
  lines.push(
    `${JSON.stringify(endpoint, null, 2)
      .split("\n")
      .map((line) => `  ${line}`)
      .join("\n")},`,
  );
}
lines.push("];");
lines.push("");

const output = lines.join("\n");

if (check) {
  const current = existsSync(outPath) ? readFileSync(outPath, "utf8") : null;
  if (current !== output) {
    console.error("stale: web/lib/contract.ts (run `make contract`)");
    process.exit(1);
  }
} else {
  writeFileSync(outPath, output);
  console.log("wrote web/lib/contract.ts");
}
