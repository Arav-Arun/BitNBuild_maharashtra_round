"""An incident-report summary of the evidence, where every claim cites step addresses.

The template narrative is built from the evidence bundle and is always available. An
LLM narrative (``gpt-oss-120b`` by default) is optional: it must return JSON claims with
``cites``, and the validator rejects any claim that cites an unknown address or states
a number that does not appear in the bundle. One failed retry falls back to the template.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
SCHEMA_HINT = (
    'Return JSON only: {"claims": [{"text": "...", "cites": ["<step address>", ...]}]}. '
    "Use 3 to 6 short claims. Cite only step addresses that appear in the evidence, and "
    "only state numbers that appear in it."
)


def _numbers(value: Any) -> set[str]:
    """Every number in the bundle, in the forms a sentence may reasonably print it."""
    found: set[str] = set()
    for token in NUMBER.findall(json.dumps(value, default=str)):
        x = float(token)
        found.add(token)
        for digits in range(0, 5):
            found.add(f"{x:.{digits}f}")
            found.add(f"{x * 100:.{digits}f}")  # a probability printed as a percentage
        if x.is_integer():
            found.add(str(int(x)))
    return found


def validate(narrative: dict[str, Any], bundle: dict[str, Any], addrs: set[str]) -> list[str]:
    """Problems with a narrative; an empty list means it is grounded in the bundle."""
    problems = []
    claims = narrative.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["no claims"]
    allowed = _numbers(bundle)
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            problems.append(f"claim {i} is not {{text, cites}}")
            continue
        cites = claim.get("cites") or []
        if not cites:
            problems.append(f"claim {i} cites nothing")
        for addr in cites:
            if addr not in addrs:
                problems.append(f"claim {i} cites unknown step {addr!r}")
        for token in NUMBER.findall(claim["text"]):
            if token not in allowed and token.lstrip("-") not in allowed:
                problems.append(f"claim {i} states {token}, which is not in the evidence")
    return problems


def template_narrative(bundle: dict[str, Any]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    suspects = bundle.get("suspects") or []
    if not suspects:
        return {"source": "template", "claims": []}
    top = suspects[0]
    addr = top["addr"]
    if bundle.get("abstain"):
        members = bundle.get("conformal_set") or []
        claims.append(
            {
                "text": f"No confident culprit: the {bundle['coverage_target']:.0%} candidate "
                f"set has {bundle['conformal_size']} steps. The strongest suspect is {addr} "
                f"({top['probability']:.0%}).",
                "cites": members or [addr],
            }
        )
    else:
        claims.append(
            {
                "text": f"The most likely root cause is {addr} ({top['probability']:.0%}).",
                "cites": [addr],
            }
        )
    for reason in top.get("reasons", [])[:2]:
        claims.append({"text": reason["text"], "cites": [addr]})
    for line in top.get("evidence", [])[:1]:
        claims.append({"text": line["text"], "cites": [line["addr"]]})
    damage = top.get("damage_path") or {}
    if damage.get("suspect_matches_twin"):
        claims.append(
            {
                "text": f"However, {addr} produced exactly what the passing run of this task "
                "produced, so the damage downstream did not start there.",
                "cites": [addr],
            }
        )
    elif damage.get("nodes"):
        symptoms = [n["addr"] for n in damage["nodes"] if n["tag"] == "symptom"]
        reach = "reaches the final answer" if damage.get("reaches_final") else "spreads"
        claims.append(
            {
                "text": f"Its output {reach} through {damage['reached']} later steps, "
                f"{damage['symptoms']} of which show damage.",
                "cites": [addr, *symptoms[:4]],
            }
        )
    divergence = bundle.get("twin_divergence")
    if divergence and divergence != addr:
        claims.append(
            {
                "text": f"Compared with the passing run of the same task, the first step to "
                f"diverge on identical input is {divergence}.",
                "cites": [divergence],
            }
        )
    precedent = top.get("precedents") or {}
    if precedent.get("summary"):
        claims.append({"text": precedent["summary"], "cites": [addr]})
    for check in bundle.get("verification") or []:
        if check.get("samples"):
            claims.append(
                {
                    "text": f"Intervention at {check['addr']}: {check['fix_rate']:.0%} of "
                    f"{check['samples']} edited replays passed vs {check['control_rate']:.0%} "
                    f"unedited, so it is {check['verdict']}.",
                    "cites": [check["addr"]],
                }
            )
        elif check.get("masked_by"):
            claims.append({"text": check["note"], "cites": [check["addr"], check["masked_by"]]})
    return {"source": "template", "claims": claims}


async def llm_narrative(
    bundle: dict[str, Any], addrs: set[str], client: Any, model: str
) -> dict[str, Any]:
    """Ask the LLM for cited claims; validate; retry once; else use the template."""
    messages = [
        {
            "role": "system",
            "content": "You write incident reports about failed AI-agent runs. " + SCHEMA_HINT,
        },
        {"role": "user", "content": json.dumps(bundle, default=str)},
    ]
    for attempt in range(2):
        try:
            response = await client.chat(
                messages=messages, model=model, response_format={"type": "json_object"}
            )
            narrative = json.loads(response["choices"][0]["message"]["content"])
        except Exception as error:  # any provider or parse failure means "no LLM narrative"
            logger.info("narrative attempt %d failed: %s", attempt + 1, error)
            continue
        problems = validate(narrative, bundle, addrs)
        if not problems:
            return {"source": "llm", "model": model, "claims": narrative["claims"]}
        messages += [
            {"role": "assistant", "content": json.dumps(narrative)},
            {"role": "user", "content": "Fix these problems: " + "; ".join(problems[:10])},
        ]
    return template_narrative(bundle)
