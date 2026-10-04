"""Convert evaluator JSON files into the typed Results-page response."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from server import models as m

SPLITS = ["S0", "S1", "S3", "S4", "S5"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def estimate(raw: dict[str, Any] | None):
    if not raw:
        return None
    ci = raw.get("ci")
    return m.Estimate(
        value=float(raw["value"]),
        ci_low=float(ci[0]) if ci else None,
        ci_high=float(ci[1]) if ci else None,
    )


def difference(raw: dict[str, Any] | None):
    if not raw or not raw.get("ci") or not raw.get("n"):
        return None
    low, high = map(float, raw["ci"])
    return m.PairedDifference(
        delta=float(raw["difference"]),
        ci_low=low,
        ci_high=high,
        n=int(raw["n"]),
        n_boot=2000,
        excludes_zero=low > 0 or high < 0,
    )


def _estimate_all(raw):
    return {key: estimate(raw.get(key)) for key in ("top1", "top3", "within1", "mrr")}


def build_eval(service):
    root = service.eval_dir
    board, ablations = load(root / "leaderboard.json"), load(root / "ablations.json")
    summary, calibration = load(root / "summary.json"), load(root / "calibration.json")
    integrity, detector = load(root / "integrity.json"), load(root / "run_detector.json")
    if not board or not summary:
        raise FileNotFoundError("Run make eval to create results artifacts.")

    raw_methods = {
        name: value
        for name, value in board.items()
        if isinstance(value, dict) and "splits" in value
    }
    method_names = [name.replace("_", "-") for name in raw_methods]
    output_rows, cells = [], []
    for name, source in raw_methods.items():
        method = name.replace("_", "-")
        for split in SPLITS:
            raw = source.get("splits", {}).get(split)
            if split == "S3":
                raw = None  # Only agent-specific S3 data exists.
            metrics = _estimate_all(raw or {})
            ran = bool(raw and all(metrics[k] and metrics[k].ci_low is not None for k in metrics))
            reason = (
                None
                if ran
                else (
                    "Evaluation is reported per agent; no aggregate was computed."
                    if split == "S3"
                    else "This method and split were not evaluated."
                )
            )
            output_rows.append(
                m.LeaderboardRow(
                    method=method,
                    method_kind="ranker" if name == "blackbox" else "baseline",
                    split=split,
                    status="ran" if ran else "not_run",
                    reason=reason,
                    reported_literature=False,
                    source=None,
                    n=int(raw["n"]) if ran else None,
                    top1=metrics["top1"] if ran else None,
                    top3=metrics["top3"] if ran else None,
                    within1=metrics["within1"] if ran else None,
                    mrr=metrics["mrr"] if ran else None,
                    latency_ms=None,
                )
            )
            cells.append(
                m.MatrixCell(
                    method=method,
                    split=split,
                    value=metrics["top1"].value if ran else None,
                    n=int(raw["n"]) if ran else None,
                    status="ran" if ran else "not_run",
                    reason=reason,
                )
            )

    measured = raw_methods["blackbox"]["splits"]
    unseen = measured.get("S1", {})
    natural = measured.get("S4", {})
    baseline = max(
        (
            (name, val.get("splits", {}).get("S1", {}))
            for name, val in raw_methods.items()
            if name != "blackbox"
        ),
        key=lambda pair: pair[1].get("top1", {}).get("value", -1),
        default=(None, {}),
    )
    metrics = summary.get("metrics", {})
    headline = []

    def card(
        identifier,
        title,
        raw,
        unit="fraction",
        n=None,
        comparator_name=None,
        comparator=None,
        tooltip="",
    ):
        est = estimate(raw)
        headline.append(
            m.HeadlineCard(
                id=identifier,
                title=title,
                status="ran" if est else "not_run",
                reason=None if est else "Metric not produced by this evaluation.",
                value=est,
                unit=unit,
                n=n,
                comparator_name=comparator_name,
                comparator=estimate(comparator),
                tooltip=tooltip,
            )
        )

    card(
        "unseen_fault_top1",
        "Unseen fault types, top-1",
        unseen.get("top1"),
        n=unseen.get("n"),
        comparator_name=baseline[0].replace("_", "-") if baseline[0] else None,
        comparator=baseline[1].get("top1"),
        tooltip="Top-1 root localization on held-out fault types.",
    )
    card(
        "natural_top1",
        "Natural failures, top-1",
        natural.get("top1"),
        n=natural.get("n"),
        tooltip="Top-1 root localization on naturally occurring failures.",
    )
    card(
        "replay_savings",
        "Calls saved by cone replay",
        {"value": metrics["replay_savings"]} if "replay_savings" in metrics else None,
        tooltip="Aggregate replay-cache savings; no per-run interval is available.",
    )
    card(
        "ms_per_trace",
        "Diagnosis latency",
        {"value": metrics["ms_per_trace"]} if "ms_per_trace" in metrics else None,
        unit="ms",
        tooltip="Mean model inference latency per trace.",
    )

    ablation_rows = []
    for title, raw in ablations.get("feature_groups", {}).items():
        values = raw.get("splits", {}).get("S1", {})
        diff = raw.get("vs_full", {}).get("S1", {})
        has = bool(values.get("top1"))
        ablation_rows.append(
            m.FeatureGroupAblationRow(
                group_id=title[1:2],
                group_name=title[3:],
                status="ran" if has else "not_run",
                reason=None if has else "Missing result.",
                top1_without=estimate(values.get("top1")),
                delta_vs_full=difference(diff),
            )
        )
    full = estimate(unseen.get("top1"))
    feature_ablation = m.FeatureGroupAblation(
        split="S1", n=unseen.get("n"), full=full, rows=ablation_rows
    )

    paired = []
    for key, ident, title, arm_defs in [
        (
            "edges",
            "provenance_edges",
            "Full provenance vs protocol-only vs no edges",
            [
                ("full", "Full provenance edges"),
                ("protocol", "Protocol-only edges"),
                ("none", "No edges"),
            ],
        ),
        (
            "observability",
            "telemetry_sufficiency",
            "Full record vs OTel-style fields vs outputs-only",
            [
                ("full", "Full Black Box record"),
                ("otel", "OTel-style fields"),
                ("outputs_only", "Outputs only"),
            ],
        ),
    ]:
        source = ablations.get(key, {})
        arms = []
        for i, (arm, label) in enumerate(arm_defs):
            values = source.get(arm, {}).get("splits", {}).get("S1", {})
            diff = source.get(arm, {}).get("vs_full", {}).get("S1") if i else None
            found = bool(values.get("top1"))
            arms.append(
                m.AblationArm(
                    arm=arm,
                    label=label,
                    status="ran" if found else "not_run",
                    reason=None if found else "Ablation was not evaluated.",
                    top1=estimate(values.get("top1")),
                    diff_vs_reference=difference(diff),
                )
            )
        diffs = [arm.diff_vs_reference for arm in arms[1:]]
        if any(x is None for x in diffs):
            interpretation = None
        elif any(x.ci_low > 0 for x in diffs if x):
            interpretation = "reference_loses"
        elif all(x.ci_high < 0 for x in diffs if x):
            interpretation = "reference_wins"
        else:
            interpretation = "no_significant_difference"
        paired.append(
            m.PairedAblation(
                id=ident,
                title=title,
                split="S1",
                n=unseen.get("n"),
                reference_arm="full",
                arms=arms,
                interpretation=interpretation,
            )
        )

    cal = calibration.get("S1", {})
    calibration_model = m.Calibration(
        split="S1",
        status="ran" if cal else "not_run",
        reason=None if cal else "Calibration artifact missing.",
        bins=[
            m.ReliabilityBin(
                lo=x["bin"][0],
                hi=x["bin"][1],
                n=x["n"],
                mean_confidence=x["confidence"],
                accuracy=x["accuracy"],
            )
            for x in cal.get("reliability", [])
        ],
        abstention=[],
        ece=cal.get("ece"),
        conformal_target=cal.get("target_coverage"),
        conformal_empirical=cal.get("coverage"),
        mean_set_size=cal.get("mean_set_size"),
        abstain_rate=cal.get("abstain_rate"),
    )

    checks = []
    shuffled = integrity.get("shuffled_labels", {})
    raw = shuffled.get("top1")
    checks.append(
        m.IntegrityCheck(
            id="shuffled_labels",
            title="Shuffled labels",
            status="ran" if raw else "not_run",
            reason=None if raw else "Artifact missing.",
            value=estimate(raw),
            threshold=None,
            passes=bool(raw and raw.get("value", 1) < 0.25),
            n=shuffled.get("n"),
            detail="Top-1 after training relevance labels are shuffled.",
        )
    )
    audit = integrity.get("artifact_audit", {})
    has = "auroc" in audit
    checks.append(
        m.IntegrityCheck(
            id="artifact_audit",
            title="Artifact audit",
            status="ran" if has else "not_run",
            reason=None if has else "Artifact missing.",
            value=m.Estimate(value=audit["auroc"], ci_low=None, ci_high=None) if has else None,
            threshold=audit.get("threshold"),
            passes=audit.get("passed"),
            n=audit.get("n_injected"),
            detail="AUROC of a classifier using formatting-only injected-step signals.",
        )
    )
    checks.append(
        m.IntegrityCheck(
            id="human_agreement",
            title="Human agreement",
            status="not_run",
            reason="This evaluation predates double-annotator labels.",
            value=None,
            threshold=None,
            passes=None,
            n=None,
            detail="Cohen's kappa from overlapping human labels.",
        )
    )

    rows = service._all_rows()
    labels = [
        row
        for agent in service.agents.values()
        for row in agent.database.query("SELECT * FROM labels")
    ]
    versions = []
    frozen = None
    for agent in service.agents.values():
        path = agent.data_dir / "frozen" / "DATASET_VERSION"
        if path.is_file():
            versions.append(path.read_text().strip())
            stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            frozen = max(frozen, stamp) if frozen else stamp
    from blackbox.api.index import FAULT_SPECS

    groups = {}
    for label in labels:
        spec = FAULT_SPECS.get(label["fault_type"])
        if spec:
            groups[spec.family] = groups.get(spec.family, 0) + 1
    dataset = m.DatasetInfo(
        version="+".join(x[:16] for x in versions)
        or str(summary.get("details", {}).get("dataset_hash", "unknown")),
        frozen_at=frozen,
        agents=sorted(service.agents),
        held_out_fault_codes=sorted({s.code for s in FAULT_SPECS.values() if s.held_out}),
        counts=m.DatasetCounts(
            base_runs=sum(r["origin"] == "natural" for r in rows),
            injected_forks=sum(r["source"] == "injected" for r in labels),
            accepted_failures=sum(
                r["origin"] == "injected" and r["outcome"] == "failed" for r in rows
            ),
            natural_failures=sum(r["source"] == "natural_auto" for r in labels),
            human_labelled=sum(r["source"] == "human" for r in labels),
            runs_per_split={s: sum(r["split"] == s for r in rows) for s in SPLITS},
            failures_per_family=groups,
        ),
    )
    bundle = service.model
    diag = bundle.diagnoser if bundle else None
    meta = getattr(diag, "metadata", {}) if diag else {}
    config = getattr(diag, "config", None)
    names = meta.get("feature_names", [])
    enabled = list(getattr(config, "groups", []))
    from blackbox.ml.features import FEATURE_GROUPS

    group_names = {
        "A": "Structure",
        "B": "Telemetry",
        "C": "Validity",
        "D": "Grounding",
        "E": "Surprisal",
        "F": "Novelty",
        "G": "Context",
        "H": "Lineage",
    }
    feature_groups = [
        m.FeatureGroupInfo(
            id=g,
            name=group_names.get(g, g),
            n_features=sum(group == g for group in FEATURE_GROUPS.values()),
            enabled=True,
        )
        for g in enabled
    ]
    module = type(getattr(diag, "booster", None)).__module__ if diag else ""
    model = m.ModelInfo(
        model_version=bundle.version if bundle else "unavailable",
        backend="lightgbm" if "lightgbm" in module else "sklearn-hgb",
        trained_at=bundle.trained_at if bundle else None,
        n_features=len(names),
        feature_groups=feature_groups,
        temperature=getattr(diag, "temperature", None),
        conformal_coverage=1 - getattr(config, "alpha", 0.1) if config else None,
        trained_on_runs=int(meta.get("n_train_runs", 0)),
    )
    detections = detector.get("n_failed", 0) + detector.get("n_healthy", 0)
    not_run = [
        m.NotRunItem(
            section="leaderboard", item="S3 aggregate", reason="Published per agent only."
        ),
        m.NotRunItem(
            section="leaderboard",
            item="S5 zero-shot",
            reason="External benchmark evaluation was not run.",
        ),
        m.NotRunItem(
            section="integrity",
            item="Human agreement",
            reason="No double-annotated runs in the frozen evaluation.",
        ),
    ]
    generated = datetime.fromtimestamp((root / "summary.json").stat().st_mtime, UTC)
    return m.EvalResponse(
        fixture=False,
        fixture_notice=None,
        generated_at=generated,
        dataset=dataset,
        model=model,
        headline=headline,
        leaderboard=output_rows,
        generalisation=m.GeneralisationMatrix(
            metric="top1", methods=method_names, splits=SPLITS, cells=cells
        ),
        feature_group_ablation=feature_ablation,
        recorder_ablations=paired,
        integrity=checks,
        calibration=calibration_model,
        replay_savings=m.ReplaySavings(
            status="not_run",
            reason="Replay-level aggregates were not written by make eval.",
            n_forks=0,
            by_mode=[],
            calls_saved_fraction=None,
            tokens_saved=0,
            ms_saved=0,
            verifier=None,
        ),
        detector=m.DetectorMetrics(
            status="ran" if detector else "not_run",
            reason=None if detector else "Detector artifact missing.",
            split="S0" if detector else None,
            n_runs=int(detections),
            n_failed=int(detector.get("n_failed", 0)),
            auroc=m.Estimate(value=detector["auroc"], ci_low=None, ci_high=None)
            if detector
            else None,
            accuracy=None,
        ),
        not_run=not_run,
        notes=[
            "Measured values are loaded from data/eval artifacts.",
            "The S3 benchmark is reported per agent; no pooled number is inferred.",
        ],
    )
