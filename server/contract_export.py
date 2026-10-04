"""Export the pydantic contract as a JSON Schema bundle and regenerate ``web/lib/contract.ts``.

    python -m server.contract_export             # write server/contract.schema.json + web/lib/contract.ts
    python -m server.contract_export --check     # fail if either committed file is stale
    python -m server.contract_export --no-ts     # schema only (no node required)

The output is a pure function of ``server/models.py`` and the pinned generator, so running it
twice never produces a diff.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from server import models

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "server" / "contract.schema.json"
WEB_DIR = ROOT / "web"
TS_PATH = WEB_DIR / "lib" / "contract.ts"

# Keys whose values are data, not subschemas.
_DATA_KEYS = {"default", "const", "enum", "examples", "required", "x-enums", "x-endpoints"}
# Keys whose values map names to subschemas.
_SCHEMA_MAPS = {"properties", "$defs", "patternProperties"}


def _clean(node: Any, *, keep_title: bool = False) -> Any:
    """Drop per-property titles (they become noisy type aliases) and make tuples TS-friendly."""
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str) and not keep_title:
            continue
        if key == "prefixItems":
            cleaned["items"] = [_clean(item) for item in value]
            continue
        if key in _DATA_KEYS:
            cleaned[key] = value
        elif key in _SCHEMA_MAPS:
            if key == "$defs":
                cleaned[key] = {name: _clean(item, keep_title=True) for name, item in value.items()}
            else:
                cleaned[key] = {name: _clean(item) for name, item in value.items()}
        else:
            cleaned[key] = _clean(value)
    return cleaned


def build_schema() -> dict[str, Any]:
    _, schema = models_json_schema(
        [(model, "validation") for model in models.EXPORTED_MODELS],
        ref_template="#/$defs/{model}",
    )
    bundle = _clean(schema)
    bundle["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    bundle["title"] = "BlackBoxContract"
    bundle["x-contract-version"] = models.CONTRACT_VERSION
    bundle["x-enums"] = {name: list(values) for name, values in models.ENUMS.items()}
    bundle["x-unions"] = {
        name: [member.__name__ for member in members] for name, members in models.UNIONS.items()
    }
    bundle["x-endpoints"] = [asdict(endpoint) for endpoint in models.ENDPOINTS]
    bundle["x-exported"] = [model.__name__ for model in models.EXPORTED_MODELS]
    return bundle


def render_schema() -> str:
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


def _npm() -> str | None:
    return shutil.which("npm")


def node_available() -> bool:
    return shutil.which("node") is not None and _npm() is not None


def run_generator(*extra: str) -> subprocess.CompletedProcess[str]:
    npm = _npm()
    if npm is None:
        raise RuntimeError("npm is required to generate web/lib/contract.ts (use --no-ts to skip)")
    return subprocess.run(
        [npm, "--prefix", str(WEB_DIR), "run", "--silent", "contract", "--", *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if committed outputs are stale")
    parser.add_argument("--no-ts", action="store_true", help="skip TypeScript generation")
    args = parser.parse_args(argv)

    schema_text = render_schema()
    if args.check:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else None
        if current != schema_text:
            print(f"stale: {SCHEMA_PATH.relative_to(ROOT)} (run `make contract`)", file=sys.stderr)
            return 1
    else:
        SCHEMA_PATH.write_text(schema_text, encoding="utf-8")
        print(f"wrote {SCHEMA_PATH.relative_to(ROOT)}")
    if args.no_ts:
        return 0

    result = run_generator("--check") if args.check else run_generator()
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
