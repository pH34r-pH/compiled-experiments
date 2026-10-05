#!/usr/bin/env python3
"""Compare a fresh #152 reproduction with the frozen historical decision.

Scientific equivalence is defined against the preregistered decision boundaries,
not against an arbitrary decimal tolerance. Exact identities and categorical
outcomes must match. Thresholded metrics must remain in the same decision cell
and retain at least half of the historical distance from the relevant boundary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

NUMERIC_RATIO_MAX = 1.15
MODULE_EXCEED_COUNT_MAX = 2
ROUTING_WIN_FRACTION_MIN = 0.75
MINIMUM_RETAINED_REFERENCE_MARGIN = 0.5

EXACT_TOP_LEVEL_FIELDS = (
    "decision",
    "validated_pair_count",
    "required_pair_count",
    "held_out_trace_count",
)
EXACT_PAIR_FIELDS = (
    "configuration_count",
    "expected_lower_cost_member",
    "numeric_eligible",
    "pair_validates",
)
CAPTURE_IDENTITY_FIELDS = (
    "model",
    "revision",
    "tokenizer_model",
    "tokenizer_revision",
    "input_ids_sha256",
    "attention_mask_sha256",
    "prompt_sha256",
    "valid_token_count",
    "trace_count",
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("issue") != "#152":
        raise ValueError(f"{path}: expected #152 summary")
    return value


def _same_side(value: float, threshold: float, *, pass_when_le: bool) -> bool:
    return value <= threshold if pass_when_le else value >= threshold


def _check_margin(
    *,
    pair_id: str,
    metric: str,
    reference: float,
    candidate: float,
    threshold: float,
    pass_when_le: bool,
    failures: list[str],
    tolerated: list[str],
) -> None:
    reference_state = _same_side(reference, threshold, pass_when_le=pass_when_le)
    candidate_state = _same_side(candidate, threshold, pass_when_le=pass_when_le)
    if candidate_state != reference_state:
        failures.append(
            f"{pair_id}.{metric}: crossed the preregistered decision boundary "
            f"{threshold!r}: expected {reference!r}, got {candidate!r}"
        )
        return

    reference_margin = abs(reference - threshold)
    candidate_margin = abs(candidate - threshold)
    if reference_margin == 0.0:
        if candidate != reference:
            failures.append(
                f"{pair_id}.{metric}: historical value lies exactly on decision boundary "
                f"{threshold!r}; candidate must match it exactly"
            )
        return

    retained = candidate_margin / reference_margin
    if retained < MINIMUM_RETAINED_REFERENCE_MARGIN:
        failures.append(
            f"{pair_id}.{metric}: retained only {retained:.6f} of the historical "
            f"decision margin; require >= {MINIMUM_RETAINED_REFERENCE_MARGIN:.3f} "
            f"(reference={reference!r}, candidate={candidate!r}, boundary={threshold!r})"
        )
    elif candidate != reference:
        tolerated.append(
            f"{pair_id}.{metric}: expected {reference!r}, got {candidate!r}; "
            f"retained {retained:.6f} of the historical decision margin"
        )


def _derived_numeric_eligible(row: dict) -> bool:
    return (
        float(row["operand_nmse_symmetric_geomean_ratio"]) <= NUMERIC_RATIO_MAX
        and float(row["dot_nmse_symmetric_geomean_ratio"]) <= NUMERIC_RATIO_MAX
        and int(row["modules_exceeding_1_35_ratio"]) <= MODULE_EXCEED_COUNT_MAX
    )


def compare(reference: dict, candidate: dict) -> dict:
    failures: list[str] = []
    tolerated: list[str] = []

    for field in EXACT_TOP_LEVEL_FIELDS:
        if candidate.get(field) != reference.get(field):
            failures.append(
                f"{field}: expected {reference.get(field)!r}, got {candidate.get(field)!r}"
            )

    ref_rows = {row["pair_id"]: row for row in reference["pair_decisions"]}
    got_rows = {row["pair_id"]: row for row in candidate["pair_decisions"]}
    if set(got_rows) != set(ref_rows):
        failures.append(
            f"pair roster differs: expected {sorted(ref_rows)}, got {sorted(got_rows)}"
        )

    for pair_id in sorted(set(ref_rows) & set(got_rows)):
        ref = ref_rows[pair_id]
        got = got_rows[pair_id]

        for field in EXACT_PAIR_FIELDS:
            if got.get(field) != ref.get(field):
                failures.append(
                    f"{pair_id}.{field}: expected {ref.get(field)!r}, got {got.get(field)!r}"
                )

        ref_derived = _derived_numeric_eligible(ref)
        got_derived = _derived_numeric_eligible(got)
        if ref_derived != bool(ref["numeric_eligible"]):
            failures.append(f"{pair_id}: historical numeric_eligible is inconsistent with frozen thresholds")
        if got_derived != bool(got["numeric_eligible"]):
            failures.append(f"{pair_id}: reproduced numeric_eligible is inconsistent with frozen thresholds")

        _check_margin(
            pair_id=pair_id,
            metric="operand_nmse_symmetric_geomean_ratio",
            reference=float(ref["operand_nmse_symmetric_geomean_ratio"]),
            candidate=float(got["operand_nmse_symmetric_geomean_ratio"]),
            threshold=NUMERIC_RATIO_MAX,
            pass_when_le=True,
            failures=failures,
            tolerated=tolerated,
        )
        _check_margin(
            pair_id=pair_id,
            metric="dot_nmse_symmetric_geomean_ratio",
            reference=float(ref["dot_nmse_symmetric_geomean_ratio"]),
            candidate=float(got["dot_nmse_symmetric_geomean_ratio"]),
            threshold=NUMERIC_RATIO_MAX,
            pass_when_le=True,
            failures=failures,
            tolerated=tolerated,
        )
        _check_margin(
            pair_id=pair_id,
            metric="modules_exceeding_1_35_ratio",
            reference=float(ref["modules_exceeding_1_35_ratio"]),
            candidate=float(got["modules_exceeding_1_35_ratio"]),
            threshold=float(MODULE_EXCEED_COUNT_MAX),
            pass_when_le=True,
            failures=failures,
            tolerated=tolerated,
        )

        count = int(ref["configuration_count"])
        if int(got["configuration_count"]) == count:
            ref_wins = round(float(ref["routing_win_fraction"]) * count)
            got_wins = round(float(got["routing_win_fraction"]) * count)
            routing_threshold_count = math.ceil(ROUTING_WIN_FRACTION_MIN * count)
            _check_margin(
                pair_id=pair_id,
                metric="routing_win_count",
                reference=float(ref_wins),
                candidate=float(got_wins),
                threshold=float(routing_threshold_count),
                pass_when_le=False,
                failures=failures,
                tolerated=tolerated,
            )

    historical = reference.get("capture_status", [])
    reproduced = candidate.get("capture_status", [])
    if len(reproduced) != 1 or reproduced[0].get("status") != "captured":
        failures.append("fresh reproduction did not record exactly one successful capture")
    if historical and reproduced:
        for field in CAPTURE_IDENTITY_FIELDS:
            if reproduced[0].get(field) != historical[0].get(field):
                failures.append(
                    f"capture.{field}: expected {historical[0].get(field)!r}, "
                    f"got {reproduced[0].get(field)!r}"
                )

    return {
        "schemaVersion": 2,
        "status": "matched" if not failures else "different",
        "scientificDecisionMatched": not failures,
        "referenceDecision": reference.get("decision"),
        "candidateDecision": candidate.get("decision"),
        "decisionStabilityPolicy": {
            "numericRatioMaximum": NUMERIC_RATIO_MAX,
            "modulesAbove1_35Maximum": MODULE_EXCEED_COUNT_MAX,
            "routingWinFractionMinimum": ROUTING_WIN_FRACTION_MIN,
            "minimumRetainedReferenceMargin": MINIMUM_RETAINED_REFERENCE_MARGIN,
            "exactTopLevelFields": list(EXACT_TOP_LEVEL_FIELDS),
            "exactPairFields": list(EXACT_PAIR_FIELDS),
            "exactCaptureIdentityFields": list(CAPTURE_IDENTITY_FIELDS),
        },
        "toleratedDifferences": tolerated,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(load(args.reference), load(args.candidate))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
