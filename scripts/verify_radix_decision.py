"""Independently check the frozen radix decision rule without executing the archive."""
from __future__ import annotations

import math

TOP_FIELDS = ("decision", "validated_pair_count", "required_pair_count", "held_out_trace_count")
PAIR_FIELDS = ("configuration_count", "expected_lower_cost_member", "numeric_eligible", "pair_validates")
CAPTURE_FIELDS = ("model", "revision", "tokenizer_model", "tokenizer_revision", "input_ids_sha256",
                  "attention_mask_sha256", "prompt_sha256", "valid_token_count", "trace_count")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("decision metrics must be finite numbers")
    return float(value)


def _within_cell(pair: str, metric: str, reference: float, actual: float,
                 boundary: float, lower_passes: bool) -> str | None:
    side = (lambda x: x <= boundary) if lower_passes else (lambda x: x >= boundary)
    if side(reference) != side(actual):
        raise ValueError(f"{pair}.{metric} crosses the frozen decision boundary")
    margin = abs(reference - boundary)
    if not margin:
        if actual != reference:
            raise ValueError(f"{pair}.{metric} leaves an exact boundary")
        return None
    retained = abs(actual - boundary) / margin
    if retained < 0.5:
        raise ValueError(f"{pair}.{metric} retains too little decision margin")
    if actual != reference:
        return (f"{pair}.{metric}: expected {reference!r}, got {actual!r}; "
                f"retained {retained:.6f} of the historical decision margin")
    return None


def verify_decision(reference: dict, candidate: dict) -> list[str]:
    """Return the exact tolerated metric differences after checking the frozen rule."""
    if reference.get("issue") != "#152" or candidate.get("issue") != "#152":
        raise ValueError("wrong experiment summary")
    if any(reference.get(k) != candidate.get(k) for k in TOP_FIELDS):
        raise ValueError("top-level decision differs")
    if any(type(candidate.get(k)) is not int for k in TOP_FIELDS[1:]):
        raise ValueError("decision counts must be integers")
    ref_rows = reference["pair_decisions"]
    got_rows = candidate["pair_decisions"]
    ref = {row["pair_id"]: row for row in ref_rows}
    got = {row["pair_id"]: row for row in got_rows}
    if len(ref) != len(ref_rows) or len(got) != len(got_rows) or set(ref) != set(got):
        raise ValueError("pair roster differs or repeats an identity")
    differences = []
    for pair in sorted(ref):
        a, b = ref[pair], got[pair]
        if any(a.get(k) != b.get(k) for k in PAIR_FIELDS):
            raise ValueError(f"{pair} categorical outcome differs")
        for row in (a, b):
            if type(row["pair_validates"]) is not bool or type(row["modules_exceeding_1_35_ratio"]) is not int:
                raise ValueError(f"{pair} has invalid categorical values")
            if any(_number(row[k]) < 0 for k in ("operand_nmse_symmetric_geomean_ratio",
                    "dot_nmse_symmetric_geomean_ratio", "modules_exceeding_1_35_ratio")):
                raise ValueError(f"{pair} has negative metrics")
            eligible = (_number(row["operand_nmse_symmetric_geomean_ratio"]) <= 1.15
                        and _number(row["dot_nmse_symmetric_geomean_ratio"]) <= 1.15
                        and _number(row["modules_exceeding_1_35_ratio"]) <= 2)
            if type(row["numeric_eligible"]) is not bool or row["numeric_eligible"] != eligible:
                raise ValueError(f"{pair} numeric eligibility differs from the frozen thresholds")
            if row["pair_validates"] != (eligible and _number(row["routing_win_fraction"]) >= 0.75):
                raise ValueError(f"{pair} validation differs from the frozen thresholds")
        checks = [("operand_nmse_symmetric_geomean_ratio", 1.15, True),
                  ("dot_nmse_symmetric_geomean_ratio", 1.15, True),
                  ("modules_exceeding_1_35_ratio", 2.0, True)]
        for metric, boundary, lower in checks:
            difference = _within_cell(pair, metric, _number(a[metric]), _number(b[metric]), boundary, lower)
            if difference is not None:
                differences.append(difference)
        count = a["configuration_count"]
        if type(count) is not int or type(b["configuration_count"]) is not int or count <= 0:
            raise ValueError("invalid configuration count")
        if any(not 0 <= _number(row["routing_win_fraction"]) <= 1 for row in (a, b)):
            raise ValueError("routing win fraction is out of range")
        wins_a = float(round(_number(a["routing_win_fraction"]) * count))
        wins_b = float(round(_number(b["routing_win_fraction"]) * count))
        difference = _within_cell(pair, "routing_win_count", wins_a, wins_b, float(math.ceil(0.75 * count)), False)
        if difference is not None:
            differences.append(difference)
    for summary in (reference, candidate):
        count = sum(row["pair_validates"] for row in summary["pair_decisions"])
        if (summary["validated_pair_count"] != count or summary["required_pair_count"] != 3
                or summary["decision"] != ("validate" if count >= 3 else "kill")):
            raise ValueError("aggregate decision differs from the frozen rule")
    old_capture, new_capture = reference["capture_status"], candidate["capture_status"]
    if len(old_capture) != 1 or len(new_capture) != 1 or new_capture[0].get("status") != "captured":
        raise ValueError("capture record missing or ambiguous")
    if any(old_capture[0].get(k) != new_capture[0].get(k) for k in CAPTURE_FIELDS):
        raise ValueError("capture identity differs")
    return differences
