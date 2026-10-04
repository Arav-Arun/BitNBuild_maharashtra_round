"""Hand-authored fixture content for endpoints whose backend does not exist yet.

Nothing here is a measurement. The model outputs (probabilities, calibration, the whole
evaluation) are illustrative numbers chosen to exercise every UI state, and every payload that
carries them says so: `fixture: true` on the payload, plus a notice the UI must show.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from server import models as m

FIXTURE_TIME = datetime(2026, 10, 4, 9, 30, tzinfo=UTC)
EVAL_NOTICE = (
    "ILLUSTRATIVE FIXTURE. Every number on this page is a hand-picked placeholder that exercises "
    "the layout. None of it was measured. Real values come from `make eval`."
)


def ranking(
    steps: list[m.StepDetail],
    weights: dict[str, float],
    conformal: set[str],
    flags: dict[str, list[m.SuspectFlag]],
    default_weight: float,
) -> list[m.Suspect]:
    """Turn illustrative weights into a normalised, sorted ranking whose probabilities sum to 1."""
    raw = {step.addr: weights.get(step.addr, default_weight) for step in steps}
    total = sum(raw.values())
    probabilities = {addr: round(value / total, 4) for addr, value in raw.items()}
    top = max(probabilities, key=lambda addr: probabilities[addr])
    probabilities[top] = round(probabilities[top] + (1 - sum(probabilities.values())), 4)
    order = sorted(steps, key=lambda s: (-probabilities[s.addr], s.seq))
    return [
        m.Suspect(
            rank=index,
            addr=step.addr,
            name=step.name,
            kind=step.kind,
            score=round(math.log(probabilities[step.addr]) + 4, 4),
            probability=probabilities[step.addr],
            in_conformal_set=step.addr in conformal,
            flags=flags.get(step.addr, []),
        )
        for index, step in enumerate(order, start=1)
    ]


def stale_fx_diagnosis(
    detail: m.RunDetail,
    *,
    twin: m.NearestTwin,
    precedents: m.Precedents,
    verification: m.Verification,
    damage: m.DamagePath,
    downstream: int,
    drift: float,
) -> m.Diagnosis:
    steps = detail.steps
    fx = next(step for step in steps if step.addr == "fx/tool#1")
    age = next(v for v in fx.rule_violations if v.rule == "stale_as_of")
    age_days = float(age.evidence["age_days"]) if age.evidence else 0.0
    weights = {
        "fx/tool#1": 0.82,
        "budget/tool#1": 0.06,
        "writer/chat#1": 0.03,
        "verifier/chat#1": 0.02,
        "checker/state#1": 0.02,
        "final/state#1": 0.01,
    }
    flags: dict[str, list[m.SuspectFlag]] = {
        "fx/tool#1": ["schema_violation", "first_anomaly", "on_damage_path"],
        "budget/tool#1": ["on_damage_path"],
        "writer/chat#1": ["on_damage_path"],
        "verifier/chat#1": ["on_damage_path"],
        "final/state#1": ["on_damage_path"],
        "checker/state#1": ["visible_failure", "on_damage_path"],
    }
    rate = fx.output["rate"]
    return m.Diagnosis(
        run_id=detail.run.run_id,
        model_version="fixture-diagnoser-0",
        computed_at=FIXTURE_TIME,
        latency_ms=12.4,
        fixture=True,
        ranking=ranking(steps, weights, {"fx/tool#1"}, flags, 0.004),
        conformal_set=["fx/tool#1"],
        coverage_target=0.9,
        abstain=False,
        abstain_reason=None,
        abstain_message=None,
        visible_failure_addr="checker/state#1",
        responsible_addr="fx/tool#1",
        reasons=[
            m.Reason(
                suspect_addr="fx/tool#1",
                text=f"The exchange rate is {int(age_days)} days older than the freshest data in "
                "this run.",
                citation=m.Citation(addr="fx/tool#1", json_pointer="/output/as_of"),
                feature=m.FeatureAttribution(
                    name="as_of_age_days", group="Validity", value=age_days, contribution=2.31
                ),
            ),
            m.Reason(
                suspect_addr="fx/tool#1",
                text=f"The rate ({rate}) is {drift:.0%} below the rate used by the nearest "
                "passing run.",
                citation=m.Citation(addr="fx/tool#1", json_pointer="/output/rate"),
                feature=m.FeatureAttribution(
                    name="value_drift_vs_passing", group="Grounding", value=drift, contribution=0.64
                ),
            ),
            m.Reason(
                suspect_addr="fx/tool#1",
                text=f"{downstream} later steps consume this output, so a wrong value reaches the "
                "final plan.",
                citation=m.Citation(addr="fx/tool#1", json_pointer="/output"),
                feature=m.FeatureAttribution(
                    name="downstream_consumers",
                    group="Lineage",
                    value=float(downstream),
                    contribution=1.12,
                ),
            ),
        ],
        rule_evidence=list(fx.rule_violations),
        damage_path=damage,
        precedents=precedents,
        nearest_twin=twin,
        proposed_fixes=[
            m.ProposedFix(
                addr="fx/tool#1",
                edit=m.ForkEdit(
                    addr="fx/tool#1", kind="patch_tool_args", value={"fresh": True}, known_good=True
                ),
                source="precedent",
                rationale="Re-fetch the rate with fresh=true. The same fix repaired most similar "
                "past failures, and the nearest passing run used a rate dated "
                f"{twin.known_good_values[1].value}.",
                confidence=0.78,
            )
        ],
        verification=verification,
        stages=m.StageRail(
            localize="done", attribute="done", propose="done", verify="done", current="verify"
        ),
        narrative=m.Narrative(
            summary=f"{detail.run.run_id} failed because the SGD to INR rate was stale. The budget "
            "step multiplied by it, the writer and verifier carried the total forward, and the "
            "checker rejected the plan.",
            claims=[
                m.NarrativeClaim(
                    text="The rate was dated months before every other tool result.",
                    cites=[m.Citation(addr="fx/tool#1", json_pointer="/output/as_of")],
                ),
                m.NarrativeClaim(
                    text="The budget total was computed from that rate.",
                    cites=[m.Citation(addr="budget/tool#1", json_pointer="/output/total_inr")],
                ),
                m.NarrativeClaim(
                    text="The checker found the final total did not match the fresh catalog.",
                    cites=[m.Citation(addr="checker/state#1", json_pointer="/state_after/check")],
                ),
            ],
            source="template",
            validated=True,
        ),
    )


def abstaining_diagnosis(
    detail: m.RunDetail, *, weights: dict[str, float], reasons: list[m.Reason]
) -> m.Diagnosis:
    steps = detail.steps
    conformal = set(weights)
    flags: dict[str, list[m.SuspectFlag]] = {
        steps[-1].addr: ["visible_failure"],
    }
    flags.update({addr: ["on_damage_path"] for addr in weights if addr not in flags})
    return m.Diagnosis(
        run_id=detail.run.run_id,
        model_version="fixture-diagnoser-0",
        computed_at=FIXTURE_TIME,
        latency_ms=9.8,
        fixture=True,
        ranking=ranking(steps, weights, conformal, flags, 0.01),
        conformal_set=[addr for addr in (s.addr for s in steps) if addr in conformal],
        coverage_target=0.9,
        abstain=True,
        abstain_reason="set_too_large",
        abstain_message=f"No confident culprit: the root cause is likely one of {len(conformal)} "
        "steps.",
        visible_failure_addr=steps[-1].addr,
        responsible_addr=None,
        reasons=reasons,
        rule_evidence=[v for step in steps for v in step.rule_violations],
        damage_path=None,
        precedents=None,
        nearest_twin=None,
        proposed_fixes=[],
        verification=m.Verification(
            verdict=None,
            status="not_started",
            fork_id=None,
            edit_addr=None,
            k=None,
            fix=None,
            control=None,
            replay_fidelity=None,
            preview=False,
            explanation="Verification starts after a suspect is chosen.",
        ),
        stages=m.StageRail(
            localize="done",
            attribute="abstained",
            propose="blocked",
            verify="pending",
            current="attribute",
        ),
        narrative=None,
    )


# ---------------------------------------------------------------------------
# Evaluation fixture
# ---------------------------------------------------------------------------

SPLITS: list[m.Split] = ["S0", "S1", "S3", "S4", "S5"]
SPLIT_N = {"S0": 120, "S1": 90, "S3": 60, "S4": 45, "S5": 30}

# method -> (kind, per-split top-1, latency ms). None means the cell was not run.
METHODS: dict[str, tuple[m.MethodKind, dict[str, float | None], float | None]] = {
    "blackbox-ranker": (
        "ranker",
        {"S0": 0.80, "S1": 0.50, "S3": 0.40, "S4": 0.30, "S5": None},
        12.0,
    ),
    "random-step": ("baseline", {"S0": 0.10, "S1": 0.10, "S3": 0.10, "S4": 0.10, "S5": None}, 0.1),
    "last-step": ("baseline", {"S0": 0.20, "S1": 0.20, "S3": 0.20, "S4": 0.20, "S5": None}, 0.1),
    "first-error": ("baseline", {"S0": 0.30, "S1": 0.30, "S3": 0.30, "S4": 0.20, "S5": None}, 0.2),
    "position-only": (
        "baseline",
        {"S0": 0.40, "S1": 0.20, "S3": 0.20, "S4": 0.10, "S5": None},
        0.5,
    ),
    "isolation-forest-max": (
        "baseline",
        {"S0": 0.30, "S1": 0.30, "S3": 0.20, "S4": 0.20, "S5": None},
        3.0,
    ),
    "llm-judge-all-at-once": ("llm_judge", {s: None for s in SPLITS}, None),
    "llm-judge-step-by-step": ("llm_judge", {s: None for s in SPLITS}, None),
    "llm-judge-binary-search": ("llm_judge", {s: None for s in SPLITS}, None),
}
NOT_RUN_REASON = {
    "llm_judge": "No Groq API key in this environment; the judge calls were not made.",
    "S5": "Who&When zero-shot is computed once, after the model is frozen.",
}


def estimate(value: float, spread: float) -> m.Estimate:
    return m.Estimate(
        value=round(value, 2),
        ci_low=round(max(0.0, value - spread), 2),
        ci_high=round(min(1.0, value + spread), 2),
    )


def leaderboard() -> list[m.LeaderboardRow]:
    rows: list[m.LeaderboardRow] = []
    for method, (kind, per_split, latency) in METHODS.items():
        for split in SPLITS:
            top1 = per_split[split]
            if top1 is None:
                reason = (
                    NOT_RUN_REASON["llm_judge"] if kind == "llm_judge" else NOT_RUN_REASON["S5"]
                )
                rows.append(
                    m.LeaderboardRow(
                        method=method,
                        method_kind=kind,
                        split=split,
                        status="not_run",
                        reason=reason,
                        reported_literature=False,
                        source=None,
                        n=None,
                        top1=None,
                        top3=None,
                        within1=None,
                        mrr=None,
                        latency_ms=None,
                    )
                )
                continue
            spread = round(0.5 / math.sqrt(SPLIT_N[split]) * 1.2, 2)
            rows.append(
                m.LeaderboardRow(
                    method=method,
                    method_kind=kind,
                    split=split,
                    status="ran",
                    reason=None,
                    reported_literature=False,
                    source=None,
                    n=SPLIT_N[split],
                    top1=estimate(top1, spread),
                    top3=estimate(min(1.0, top1 + 0.2), spread),
                    within1=estimate(min(1.0, top1 + 0.1), spread),
                    mrr=estimate(min(1.0, top1 + 0.15), spread),
                    latency_ms=latency,
                )
            )
    rows.append(
        m.LeaderboardRow(
            method="literature-pipeline-placeholder",
            method_kind="literature",
            split="S5",
            status="not_run",
            reason="A published number that this repository did not reproduce.",
            reported_literature=True,
            source="Placeholder row: replace with a cited value from the source ledger.",
            n=None,
            top1=m.Estimate(value=0.35, ci_low=None, ci_high=None),
            top3=None,
            within1=None,
            mrr=None,
            latency_ms=None,
        )
    )
    return rows


def paired(delta: float, half: float, n: int) -> m.PairedDifference:
    low, high = round(delta - half, 3), round(delta + half, 3)
    return m.PairedDifference(
        delta=delta,
        ci_low=low,
        ci_high=high,
        n=n,
        n_boot=2000,
        excludes_zero=low > 0 or high < 0,
    )


def ablation(
    ablation_id: str,
    title: str,
    reference: str,
    arms: list[tuple[str, str, float, float]],
    n: int,
) -> m.PairedAblation:
    """arms: (name, label, top1, delta vs reference); the first arm is the reference."""
    built = []
    for index, (name, label, top1, delta) in enumerate(arms):
        built.append(
            m.AblationArm(
                arm=name,
                label=label,
                status="ran",
                reason=None,
                top1=estimate(top1, 0.1),
                diff_vs_reference=None if index == 0 else paired(delta, 0.06, n),
            )
        )
    others = [arm.diff_vs_reference for arm in built if arm.diff_vs_reference]
    if any(d.ci_low > 0 for d in others):
        verdict = "reference_loses"
    elif others and all(d.ci_high < 0 for d in others):
        verdict = "reference_wins"
    else:
        verdict = "no_significant_difference"
    return m.PairedAblation(
        id=ablation_id,
        title=title,
        split="S1",
        n=n,
        reference_arm=reference,
        arms=built,
        interpretation=verdict,
    )


GROUPS = [
    ("A", "Structure", 14),
    ("B", "Telemetry", 12),
    ("C", "Validity", 12),
    ("D", "Grounding", 10),
    ("E", "Surprisal", 8),
    ("F", "Novelty", 8),
    ("G", "Context", 12),
    ("H", "Lineage", 10),
    ("I", "Embedding", 4),
]


def evaluation() -> m.EvalResponse:
    board = leaderboard()
    ranker_s1 = next(r for r in board if r.method == "blackbox-ranker" and r.split == "S1")
    ranker_s4 = next(r for r in board if r.method == "blackbox-ranker" and r.split == "S4")
    best_baseline = max(
        (r for r in board if r.method_kind == "baseline" and r.split == "S1" and r.top1),
        key=lambda r: r.top1.value,  # type: ignore[union-attr]
    )
    methods = list(METHODS)
    matrix = [
        m.MatrixCell(
            method=row.method,
            split=row.split,
            value=row.top1.value if row.top1 else None,
            n=row.n,
            status=row.status,
            reason=row.reason,
        )
        for row in board
        if row.method in METHODS
    ]
    group_rows = []
    for index, (gid, name, _) in enumerate(GROUPS):
        if gid == "E":
            group_rows.append(
                m.FeatureGroupAblationRow(
                    group_id=gid,
                    group_name=name,
                    status="not_run",
                    reason="Qwen3-0.6B surprisal needs a GPU and was not computed here.",
                    top1_without=None,
                    delta_vs_full=None,
                )
            )
            continue
        delta = round(-0.02 * (index % 5 + 1), 3)
        group_rows.append(
            m.FeatureGroupAblationRow(
                group_id=gid,
                group_name=name,
                status="ran",
                reason=None,
                top1_without=estimate(0.5 + delta, 0.1),
                delta_vs_full=paired(delta, 0.04, 90),
            )
        )
    bins = [
        m.ReliabilityBin(
            lo=round(i / 10, 1),
            hi=round((i + 1) / 10, 1),
            n=10 + i,
            mean_confidence=round(i / 10 + 0.05, 2),
            accuracy=round(min(1.0, i / 10 + 0.02), 2),
        )
        for i in range(10)
    ]
    return m.EvalResponse(
        fixture=True,
        fixture_notice=EVAL_NOTICE,
        generated_at=FIXTURE_TIME,
        dataset=m.DatasetInfo(
            version="fixture-0000",
            frozen_at=None,
            agents=["tripcrew", "hoprag"],
            held_out_fault_codes=["T2", "T5", "R2", "D3", "C2", "C3"],
            counts=m.DatasetCounts(
                base_runs=100,
                injected_forks=400,
                accepted_failures=300,
                natural_failures=45,
                human_labelled=0,
                runs_per_split={s: SPLIT_N[s] for s in SPLITS},
                failures_per_family={
                    "tool": 100,
                    "retrieval": 50,
                    "decision": 80,
                    "coordination": 70,
                },
            ),
        ),
        model=m.ModelInfo(
            model_version="fixture-diagnoser-0",
            backend="sklearn-hgb",
            trained_at=None,
            n_features=sum(n for _, _, n in GROUPS),
            feature_groups=[
                m.FeatureGroupInfo(id=g, name=name, n_features=n, enabled=g != "E")
                for g, name, n in GROUPS
            ],
            temperature=1.0,
            conformal_coverage=0.9,
            trained_on_runs=200,
        ),
        headline=[
            m.HeadlineCard(
                id="unseen_fault_top1",
                title="Unseen fault types, top-1",
                status="ran",
                reason=None,
                value=ranker_s1.top1,
                unit="fraction",
                n=ranker_s1.n,
                comparator_name=best_baseline.method,
                comparator=best_baseline.top1,
                tooltip="Top-1 on faults of types held out from training, against the best baseline.",
            ),
            m.HeadlineCard(
                id="natural_top1",
                title="Natural failures, top-1",
                status="ran",
                reason=None,
                value=ranker_s4.top1,
                unit="fraction",
                n=ranker_s4.n,
                comparator_name=None,
                comparator=None,
                tooltip="Top-1 on failures nobody injected. Test-only.",
            ),
            m.HeadlineCard(
                id="replay_savings",
                title="Calls saved by cone replay",
                status="ran",
                reason=None,
                value=estimate(0.70, 0.05),
                unit="fraction",
                n=40,
                comparator_name=None,
                comparator=None,
                tooltip="Share of LLM and tool calls served from the recording instead of re-run.",
            ),
            m.HeadlineCard(
                id="ms_per_trace",
                title="Diagnosis latency",
                status="ran",
                reason=None,
                value=m.Estimate(value=12.0, ci_low=None, ci_high=None),
                unit="ms",
                n=None,
                comparator_name=None,
                comparator=None,
                tooltip="Milliseconds to rank every step of one trace on a laptop CPU.",
            ),
        ],
        leaderboard=board,
        generalisation=m.GeneralisationMatrix(
            metric="top1", methods=methods, splits=SPLITS, cells=matrix
        ),
        feature_group_ablation=m.FeatureGroupAblation(
            split="S1", n=90, full=ranker_s1.top1, rows=group_rows
        ),
        recorder_ablations=[
            ablation(
                "provenance_edges",
                "Full provenance vs protocol-only vs no edges",
                "full_edges",
                [
                    ("full_edges", "Full provenance edges", 0.50, 0.0),
                    ("protocol_edges", "Protocol-only edges", 0.46, -0.04),
                    ("no_edges", "No edges", 0.41, -0.09),
                ],
                90,
            ),
            ablation(
                "telemetry_sufficiency",
                "Full record vs OTel-style fields vs outputs-only",
                "full_record",
                [
                    ("full_record", "Full Black Box record", 0.50, 0.0),
                    ("otel_style", "OTel-style fields", 0.38, -0.12),
                    ("outputs_only", "Outputs only", 0.30, -0.20),
                ],
                90,
            ),
        ],
        integrity=[
            m.IntegrityCheck(
                id="artifact_audit",
                title="Artifact audit",
                status="ran",
                reason=None,
                value=estimate(0.60, 0.08),
                threshold=0.65,
                passes=True,
                n=90,
                detail="AUROC of a surface-only classifier that tries to spot injected steps.",
            ),
            m.IntegrityCheck(
                id="shuffled_labels",
                title="Shuffled labels",
                status="ran",
                reason=None,
                value=estimate(0.10, 0.06),
                threshold=None,
                passes=True,
                n=90,
                detail="Top-1 after shuffling training labels; chance level is about 0.10.",
            ),
            m.IntegrityCheck(
                id="human_agreement",
                title="Human agreement (kappa)",
                status="not_run",
                reason="No run has two annotators yet.",
                value=None,
                threshold=None,
                passes=None,
                n=None,
                detail="Cohen's kappa over runs labelled in Label mode.",
            ),
        ],
        calibration=m.Calibration(
            split="S0",
            status="ran",
            reason=None,
            bins=bins,
            abstention=[
                m.AbstentionPoint(answered_fraction=f, top1_when_answered=a, n=120)
                for f, a in [(1.0, 0.80), (0.8, 0.85), (0.6, 0.90), (0.4, 0.95)]
            ],
            ece=0.05,
            conformal_target=0.9,
            conformal_empirical=0.9,
            mean_set_size=2.0,
            abstain_rate=0.2,
        ),
        replay_savings=m.ReplaySavings(
            status="ran",
            reason=None,
            n_forks=40,
            by_mode=[
                m.ModeSavings(
                    mode="cone", mean_steps_reexecuted=4.0, mean_fraction_reexecuted=0.25, n=40
                ),
                m.ModeSavings(
                    mode="prefix", mean_steps_reexecuted=8.0, mean_fraction_reexecuted=0.50, n=40
                ),
                m.ModeSavings(
                    mode="full", mean_steps_reexecuted=16.0, mean_fraction_reexecuted=1.0, n=40
                ),
            ],
            calls_saved_fraction=estimate(0.70, 0.05),
            tokens_saved=100000,
            ms_saved=50000.0,
            verifier=m.VerifierStats(
                n_known_roots=30,
                verified=20,
                refuted=2,
                inconclusive=8,
                n_wrong_roots=20,
                falsely_verified=0,
            ),
        ),
        detector=m.DetectorMetrics(
            status="ran",
            reason=None,
            split="S0",
            n_runs=200,
            n_failed=100,
            auroc=estimate(0.85, 0.05),
            accuracy=estimate(0.80, 0.05),
        ),
        not_run=[
            m.NotRunItem(
                section="leaderboard",
                item="LLM judge baselines",
                reason="No Groq API key in this environment.",
            ),
            m.NotRunItem(
                section="leaderboard",
                item="Who&When zero-shot (S5)",
                reason="Computed once, after the model is frozen.",
            ),
            m.NotRunItem(
                section="features", item="Surprisal group (E)", reason="Needs a GPU for Qwen3-0.6B."
            ),
            m.NotRunItem(
                section="model",
                item="LightGBM ranker",
                reason="LightGBM cannot load on this machine; sklearn-hgb is used instead.",
            ),
            m.NotRunItem(
                section="integrity",
                item="Human agreement (kappa)",
                reason="No double-annotated runs yet.",
            ),
        ],
        notes=[
            "Fixture: the structure and states are real, the numbers are not.",
            "Rows that did not run are listed explicitly rather than left blank.",
        ],
    )
