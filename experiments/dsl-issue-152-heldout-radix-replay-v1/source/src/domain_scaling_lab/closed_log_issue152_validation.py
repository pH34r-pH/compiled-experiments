"""Frozen held-out validation for GitHub issue #152.

The original q<=16 sweep found variation in product-address occupancy among
radices with the same CurveFP phase count H.  This module does not resweep or
search a held-out model.  It loads the checked-in preregistration, captures one
independent pinned pretrained model, quantizes only the frozen pairs, and
scores a fixed family of explicit routing/write/tag proxies.

CurveFP's product algebra, signed histogram, and Kulisch reduction remain
prior-art reproductions.  The only question here is whether a workload-aware
*radix-selection objective* contributes stable information beyond H.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .closed_log import ClosedLogSpec
from .closed_log_analysis import (
    AddressEnvelope,
    BankCostAnalysis,
    BankTopology,
    TraceAnalysis,
    banked_routing_tag_cost,
    routing_cost_proxy_from_components,
)
from .closed_log_experiments import evaluate_closed_log_trace, write_rows_csv
from .closed_log_traces import OperandTrace, capture_pretrained_linear_traces, load_operand_traces, save_operand_traces


DEFAULT_PREREGISTRATION = Path("results/closed_log/issue_152_validation/preregistration.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_root(preregistration: Path) -> Path:
    # .../repo/results/closed_log/issue_152_validation/preregistration.json
    return preregistration.resolve().parents[3]


def _validate_preregistration_document(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen #152 roster and decision-defining structure."""
    if document.get("issue") != "#152" or document.get("schema_version") != 1:
        raise ValueError("not a supported #152 preregistration")
    pairs = list(document["frozen_pairs"])
    if len(pairs) != 4:
        raise ValueError("#152 preregistration must freeze exactly four primary pairs")
    for pair in pairs:
        members = list(pair["members"])
        if len(members) != 2:
            raise ValueError("each frozen pair must contain exactly two radices")
        specs = [
            ClosedLogSpec(
                E=int(pair["E"]),
                C=int(pair["C"]),
                p=int(member["p"]),
                q=int(member["q"]),
                scale_bits=int(document["quantization"]["scale_bits"]),
                scale_policy=str(pair["scale_policy"]),
            )
            for member in members
        ]
        if specs[0].H != specs[1].H or specs[0].H != int(pair["H"]):
            raise ValueError(f"frozen pair {pair['id']} is not H-matched")
        if specs[0].p == specs[1].p and specs[0].q == specs[1].q:
            raise ValueError(f"frozen pair {pair['id']} duplicates a radix")
        lower = dict(pair["expected_lower_cost_member"])
        if (int(lower["p"]), int(lower["q"])) not in {(spec.p, spec.q) for spec in specs}:
            raise ValueError(f"frozen pair {pair['id']} has an unknown expected lower-cost member")
    return dict(document)


def load_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, object]:
    """Load the historical protocol and verify its original source-sweep bytes."""
    document = _validate_preregistration_document(json.loads(path.read_text()))
    source = dict(document["source_sweep"])
    root = _repo_root(path)
    source_dir = root / str(source["directory"])
    expected = {
        "152_155_rational_aggregate.csv": str(source["aggregate_csv_sha256"]),
        "152_155_rational_product_addresses.csv": str(source["product_csv_sha256"]),
    }
    for filename, digest in expected.items():
        if _sha256(source_dir / filename) != digest:
            raise ValueError(f"source sweep checksum changed: {filename}")
    return document


def load_preregistration_snapshot(path: Path, expected_sha256: str) -> dict[str, object]:
    """Load a content-pinned frozen protocol when exploratory source bytes are absent.

    The held-out #152 computation uses the already-frozen pair roster and
    thresholds; source-sweep CSVs were an historical anti-tamper guard for
    selection, not an input to held-out scoring. Portable reproductions may
    therefore replace that unavailable guard only with the exact preregistration
    byte identity plus the same structural validation.
    """
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("expected preregistration SHA-256 must be lowercase hexadecimal")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"preregistration SHA-256 changed: expected {expected_sha256}, got {actual}"
        )
    return _validate_preregistration_document(json.loads(path.read_text()))


def _candidate_name(member: Mapping[str, object]) -> str:
    return f"2^({int(member['p'])}/{int(member['q'])})"


def _candidate_spec(pair: Mapping[str, object], member: Mapping[str, object], preregistration: Mapping[str, object]) -> ClosedLogSpec:
    quantization = dict(preregistration["quantization"])
    return ClosedLogSpec(
        E=int(pair["E"]),
        C=int(pair["C"]),
        p=int(member["p"]),
        q=int(member["q"]),
        scale_bits=int(quantization["scale_bits"]),
        scale_policy=str(pair["scale_policy"]),
    )


def _required_module_suffixes(preregistration: Mapping[str, object]) -> tuple[str, ...]:
    capture = dict(preregistration["held_out_capture"])
    modules = dict(capture["modules"])
    return tuple(str(value) for value in modules["suffixes"])


def _validate_second_model_traces(traces: Sequence[OperandTrace], preregistration: Mapping[str, object]) -> list[OperandTrace]:
    capture = dict(preregistration["held_out_capture"])
    expected_model, expected_revision = str(capture["model"]), str(capture["revision"])
    expected_layers = {int(item) for item in dict(capture["modules"])["layers"]}
    suffixes = _required_module_suffixes(preregistration)
    selected = [
        trace
        for trace in traces
        if trace.source == "pretrained_transformer"
        and trace.metadata.get("model") == expected_model
        and trace.metadata.get("revision") == expected_revision
    ]
    required = {(layer, suffix) for layer in expected_layers for suffix in suffixes}
    present = {
        (int(trace.metadata.get("layer")), suffix)
        for trace in selected
        for suffix in suffixes
        if str(trace.metadata.get("module", "")).endswith(suffix)
    }
    missing = sorted(required - present)
    if missing or len(selected) != len(required):
        raise RuntimeError(
            "held-out #152 capture is incomplete; expected exactly eight real pinned GEMM traces, "
            f"missing={missing}, found={len(selected)}"
        )
    if any(not trace.metadata.get("valid_tokens_only", False) for trace in selected):
        raise RuntimeError("held-out capture did not record valid-token-only operand rows")
    return sorted(selected, key=lambda trace: (int(trace.metadata.get("layer", -1)), str(trace.metadata.get("module"))))


def ensure_second_model_traces(
    trace_directory: Path,
    preregistration: Mapping[str, object],
    *,
    capture: bool,
) -> tuple[list[OperandTrace], dict[str, object], str]:
    """Reuse or explicitly capture the required independent model—never fall back."""
    manifest_path = trace_directory / "manifest.json"
    if manifest_path.exists():
        traces, manifest = load_operand_traces(trace_directory)
        return _validate_second_model_traces(traces, preregistration), manifest, "reused"
    if not capture:
        raise RuntimeError("held-out trace directory is absent; pass --capture to attempt the pinned model capture")
    capture_spec = dict(preregistration["held_out_capture"])
    modules = dict(capture_spec["modules"])
    traces, statuses = capture_pretrained_linear_traces(
        model_specs=((str(capture_spec["model"]), str(capture_spec["revision"])),),
        prompts=tuple(str(prompt) for prompt in capture_spec["prompts"]),
        rows_per_module=int(modules["rows_per_module"]),
        outputs_per_module=int(modules["outputs_per_module"]),
        fallback_tokenizer_spec=(str(capture_spec["tokenizer"]), str(capture_spec["tokenizer_revision"])),
    )
    manifest = save_operand_traces(traces, statuses, trace_directory)
    return _validate_second_model_traces(traces, preregistration), manifest, "captured"


def _topologies(preregistration: Mapping[str, object]) -> Iterable[BankTopology]:
    grid = dict(preregistration["bank_cost_grid"])
    tagged = dict(grid["tagged_lane_merge"])
    for architecture in grid["architectures"]:
        for lane_width in grid["lane_widths"]:
            for bank_count in grid["bank_counts"]:
                for ports in grid["write_ports_per_bank"]:
                    for bank_map in grid["bank_maps"]:
                        for lane_order in grid["lane_orders"]:
                            yield BankTopology(
                                lane_width=int(lane_width),
                                bank_count=int(bank_count),
                                write_ports_per_bank=int(ports),
                                bank_map=str(bank_map),
                                lane_order=str(lane_order),
                                architecture=str(architecture),
                                sets_per_bank=int(tagged["sets_per_bank"]),
                                ways_per_set=int(tagged["ways_per_set"]),
                            )


def _topology_key(topology: BankTopology, cost_preset: str) -> tuple[object, ...]:
    return (
        topology.architecture,
        topology.lane_width,
        topology.bank_count,
        topology.write_ports_per_bank,
        topology.bank_map,
        topology.lane_order,
        topology.sets_per_bank,
        topology.ways_per_set,
        cost_preset,
    )


def _direct_topology_key(topology: BankTopology) -> tuple[object, ...]:
    """Key the routing part of a topology, independent of tag buffering."""
    return (
        topology.lane_width,
        topology.bank_count,
        topology.write_ports_per_bank,
        topology.bank_map,
        topology.lane_order,
    )


def _tag_cache_key(topology: BankTopology) -> tuple[object, ...]:
    """Tag state is independent of vector lane width and bank port count."""
    return (topology.bank_count, topology.bank_map, topology.lane_order, topology.sets_per_bank, topology.ways_per_set)


def _merge_tagged_components(
    routing_components: Mapping[str, object], tag_components: Mapping[str, object]
) -> dict[str, object]:
    """Combine shared ingress routing with cached dot-scoped tag behavior."""
    merged = dict(routing_components)
    for key in (
        "counter_writes",
        "main_histogram_flush_writes",
        "storage_bits_proxy",
        "tag_compare_bit_ops",
        "tag_allocations",
        "tag_evictions",
        "tag_local_updates",
    ):
        merged[key] = tag_components[key]
    return merged


def _summary_from_components(
    base: Mapping[str, object], components: Sequence[Mapping[str, object]], *, topology: BankTopology
) -> dict[str, object]:
    """Make CSV-friendly aggregate fields after cache-safe component merging."""
    summary = dict(base)
    summary.update(
        {
            "architecture": topology.architecture,
            "lane_width": topology.lane_width,
            "bank_count": topology.bank_count,
            "write_ports_per_bank": topology.write_ports_per_bank,
            "bank_map": topology.bank_map,
            "lane_order": topology.lane_order,
            "sets_per_bank": topology.sets_per_bank,
            "ways_per_set": topology.ways_per_set,
        }
    )
    numeric_keys = {
        key
        for component in components
        for key, value in component.items()
        if key != "dot" and isinstance(value, (int, float, np.integer, np.floating))
    }
    dots = max(1, len(components))
    for key in numeric_keys:
        total = float(sum(float(component[key]) for component in components))
        summary[f"total_{key}"] = total
        summary[f"mean_{key}_per_dot"] = total / dots
    return summary


def _geomean(values: Sequence[float]) -> float:
    safe = np.maximum(np.asarray(values, dtype=np.float64), 1e-30)
    return float(np.exp(np.mean(np.log(safe))))


def _bootstrap_upper_module_cluster(deltas: Sequence[float], *, seed: int, samples: int = 2_000) -> tuple[float, float]:
    """Deterministic paired-module bootstrap, preserving layer/role clusters."""
    data = np.asarray(deltas, dtype=np.float64)
    if not len(data):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _numeric_pair_summary(
    pair: Mapping[str, object], rows: Sequence[Mapping[str, object]], *, expected_member: str) -> dict[str, object]:
    members = {_candidate_name(member) for member in pair["members"]}
    by_member: dict[str, list[Mapping[str, object]]] = {member: [] for member in members}
    for row in rows:
        if row["pair_id"] == pair["id"]:
            by_member[str(row["candidate"])].append(row)
    if any(len(items) != 8 for items in by_member.values()):
        raise RuntimeError(f"unexpected held-out module count for {pair['id']}")
    ordered = sorted(by_member)
    a, b = by_member[ordered[0]], by_member[ordered[1]]
    # Match traces by ID before forming a paired numerical comparison.
    a_by_trace = {str(item["trace_id"]): item for item in a}
    b_by_trace = {str(item["trace_id"]): item for item in b}
    if set(a_by_trace) != set(b_by_trace):
        raise RuntimeError("candidate traces do not align")
    operand_ratios: list[float] = []
    dot_ratios: list[float] = []
    worst_module_count = 0
    for trace_id in sorted(a_by_trace):
        left, right = a_by_trace[trace_id], b_by_trace[trace_id]
        operand_a = math.sqrt(max(float(left["lhs_scalar_nmse"]), 1e-30) * max(float(left["rhs_scalar_nmse"]), 1e-30))
        operand_b = math.sqrt(max(float(right["lhs_scalar_nmse"]), 1e-30) * max(float(right["rhs_scalar_nmse"]), 1e-30))
        dot_a, dot_b = max(float(left["quantized_dot_nmse"]), 1e-30), max(float(right["quantized_dot_nmse"]), 1e-30)
        operand_ratio = max(operand_a / operand_b, operand_b / operand_a)
        dot_ratio = max(dot_a / dot_b, dot_b / dot_a)
        operand_ratios.append(operand_ratio)
        dot_ratios.append(dot_ratio)
        worst_module_count += int(max(operand_ratio, dot_ratio) > 1.35)
    eligibility = _geomean(operand_ratios) <= 1.15 and _geomean(dot_ratios) <= 1.15 and worst_module_count <= 2
    return {
        "pair_id": pair["id"],
        "expected_lower_cost_member": expected_member,
        "operand_nmse_symmetric_geomean_ratio": _geomean(operand_ratios),
        "dot_nmse_symmetric_geomean_ratio": _geomean(dot_ratios),
        "modules_exceeding_1_35_ratio": worst_module_count,
        "numeric_eligible": eligibility,
    }


def _evaluate_numeric_pair(
    pair: Mapping[str, object],
    members: Sequence[Mapping[str, object]],
    specs: Sequence[ClosedLogSpec],
    envelope: AddressEnvelope,
    traces: Sequence[OperandTrace],
    group_size: int,
) -> tuple[dict[tuple[str, str], object], list[dict[str, object]]]:
    analyses: dict[tuple[str, str], object] = {}
    rows: list[dict[str, object]] = []
    expected = (
        int(dict(pair["expected_lower_cost_member"])["p"]),
        int(dict(pair["expected_lower_cost_member"])["q"]),
    )
    for member, spec in zip(members, specs, strict=True):
        candidate = _candidate_name(member)
        for trace in traces:
            analysis, row, _, _ = evaluate_closed_log_trace(
                trace,
                spec=spec,
                group_size=group_size,
                label=f"issue152-{pair['id']}-{candidate}",
                classification="held_out_engineering_radix_selection_validation",
                issue="#152",
                validate_kulisch=True,
                compute_exact_reachability=False,
                compute_temporal_locality=False,
            )
            if (
                float(row["histogram_vs_immediate_max_abs_gap"]) > 1e-10
                or float(row["kulisch_vs_immediate_max_abs_gap"]) > 1e-10
            ):
                raise RuntimeError("exact CurveFP accumulation reproduction failed on held-out trace")
            rows.append(
                {
                    "pair_id": pair["id"],
                    "candidate": candidate,
                    "is_expected_lower_cost_member": (spec.p, spec.q) == expected,
                    **row,
                    **envelope.metadata(),
                }
            )
            analyses[(candidate, trace.trace_id)] = analysis
    return analyses, rows


def _scored_cost_rows(
    pair_id: object,
    candidate: str,
    trace: OperandTrace,
    topology: BankTopology,
    base_summary: Mapping[str, object],
    components: Sequence[Mapping[str, object]],
    cost_presets: Sequence[str],
) -> tuple[list[dict[str, object]], dict[tuple[object, ...], float]]:
    rows: list[dict[str, object]] = []
    costs: dict[tuple[object, ...], float] = {}
    for preset in cost_presets:
        scores = [
            routing_cost_proxy_from_components(
                component,
                architecture=topology.architecture,
                cost_preset=preset,
            )
            for component in components
        ]
        mean_score = float(np.mean(scores))
        summary = _summary_from_components(base_summary, components, topology=topology)
        summary.update(
            {
                "pair_id": pair_id,
                "candidate": candidate,
                "trace_id": trace.trace_id,
                "module": trace.metadata.get("module"),
                "layer": trace.metadata.get("layer"),
                "cost_preset": preset,
                "mean_total_cost_proxy_per_dot": mean_score,
                "total_total_cost_proxy": float(np.sum(scores)),
                "classification": "engineering_multi_bank_routing_write_tag_cost_proxy",
            }
        )
        rows.append(summary)
        costs[(pair_id, candidate, trace.trace_id, _topology_key(topology, preset))] = mean_score
    return rows, costs


def _pair_decision(
    pair: Mapping[str, object],
    numeric_rows: Sequence[Mapping[str, object]],
    costs: Mapping[tuple[object, ...], float],
    topologies: Sequence[BankTopology],
    cost_presets: Sequence[str],
    traces: Sequence[OperandTrace],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    members = list(pair["members"])
    expected = _candidate_name(pair["expected_lower_cost_member"])
    other = next(_candidate_name(member) for member in members if _candidate_name(member) != expected)
    numeric = _numeric_pair_summary(pair, numeric_rows, expected_member=expected)
    configuration_rows: list[dict[str, object]] = []
    for topology in topologies:
        for preset in cost_presets:
            key = _topology_key(topology, str(preset))
            expected_costs = [costs[(pair["id"], expected, trace.trace_id, key)] for trace in traces]
            other_costs = [costs[(pair["id"], other, trace.trace_id, key)] for trace in traces]
            delta = np.asarray(expected_costs) - np.asarray(other_costs)
            baseline = max(float(np.mean(other_costs)), 1e-30)
            low, high = _bootstrap_upper_module_cluster(
                delta,
                seed=int.from_bytes(
                    hashlib.sha256(f"{pair['id']}|{key}".encode()).digest()[:8],
                    "little",
                ),
            )
            module_wins = int(np.sum(delta < 0))
            row = {
                "pair_id": pair["id"],
                "expected_lower_cost_member": expected,
                "other_member": other,
                "mean_expected_cost": float(np.mean(expected_costs)),
                "mean_other_cost": float(np.mean(other_costs)),
                "mean_cost_delta_expected_minus_other": float(np.mean(delta)),
                "mean_cost_delta_fraction": float(np.mean(delta) / baseline),
                "modules_with_lower_expected_cost": module_wins,
                "bootstrap_delta_ci_low": low,
                "bootstrap_delta_ci_high": high,
                "routing_win": bool(np.mean(delta) / baseline <= -0.05 and module_wins >= 6 and high < 0),
                "architecture": topology.architecture,
                "lane_width": topology.lane_width,
                "bank_count": topology.bank_count,
                "write_ports_per_bank": topology.write_ports_per_bank,
                "bank_map": topology.bank_map,
                "lane_order": topology.lane_order,
                "cost_preset": str(preset),
            }
            configuration_rows.append(row)
    win_fraction = float(np.mean([row["routing_win"] for row in configuration_rows])) if configuration_rows else 0.0
    decision = {
        **numeric,
        "configuration_count": len(configuration_rows),
        "routing_win_fraction": win_fraction,
        "pair_validates": bool(numeric["numeric_eligible"] and win_fraction >= 0.75),
    }
    return numeric, configuration_rows, decision


def _direct_routing_components(
    analysis: TraceAnalysis,
    spec: ClosedLogSpec,
    envelope: AddressEnvelope,
    topologies: Sequence[BankTopology],
) -> dict[tuple[object, ...], BankCostAnalysis]:
    return {
        _direct_topology_key(topology): banked_routing_tag_cost(
            analysis,
            spec,
            topology,
            envelope,
            cost_preset="traffic_only",
        )
        for topology in topologies
    }


def _tagged_component_cache(
    analysis: TraceAnalysis,
    spec: ClosedLogSpec,
    envelope: AddressEnvelope,
    topologies: Sequence[BankTopology],
    *,
    lane_width: int,
    write_ports_per_bank: int,
) -> dict[tuple[object, ...], BankCostAnalysis]:
    cached: dict[tuple[object, ...], BankCostAnalysis] = {}
    for topology in topologies:
        tag_key = _tag_cache_key(topology)
        if tag_key in cached:
            continue
        canonical = BankTopology(
            lane_width=lane_width,
            bank_count=topology.bank_count,
            write_ports_per_bank=write_ports_per_bank,
            bank_map=topology.bank_map,
            lane_order=topology.lane_order,
            architecture="tagged_lane_merge",
            sets_per_bank=topology.sets_per_bank,
            ways_per_set=topology.ways_per_set,
        )
        cached[tag_key] = banked_routing_tag_cost(
            analysis,
            spec,
            canonical,
            envelope,
            cost_preset="traffic_only",
        )
    return cached


def _validation_summary(
    preregistration_path: Path,
    output: Path,
    trace_mode: str,
    manifest: Mapping[str, object],
    traces: Sequence[OperandTrace],
    pair_decisions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    surviving = [row for row in pair_decisions if row["pair_validates"]]
    final_decision = "validate" if len(surviving) >= 3 else "kill"
    return {
        "issue": "#152",
        "classification": "held_out_engineering_radix_selection_policy_test",
        "curvefp_prior_art": "CurveFP v2 signed-histogram/Kulisch semantics are reproduced; banking, tag buffers, and cost models are engineering proxies.",
        "decision": final_decision,
        "validated_pair_count": len(surviving),
        "required_pair_count": 3,
        "trace_mode": trace_mode,
        "held_out_trace_count": len(traces),
        "held_out_trace_manifest": str((output / "traces" / "manifest.json")),
        "capture_status": manifest.get("capture_status"),
        "preregistration": str(preregistration_path),
        "pair_decisions": pair_decisions,
        "cost_note": "All routing/write/tag values are explicit operation/storage/traffic proxies, not a hardware PPA result.",
    }


def _write_validation_artifacts(
    output: Path,
    numeric_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
    pair_cost_rows: Sequence[Mapping[str, object]],
    pair_numeric_rows: Sequence[Mapping[str, object]],
    pair_decisions: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    write_rows_csv(output / "heldout_numeric.csv", numeric_rows)
    write_rows_csv(output / "bank_cost_by_module.csv", cost_rows)
    write_rows_csv(output / "bank_cost_pair_deltas.csv", pair_cost_rows)
    write_rows_csv(output / "pair_numeric_eligibility.csv", pair_numeric_rows)
    write_rows_csv(output / "pair_decisions.csv", pair_decisions)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )


def run_issue152_validation(
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    output: Path = Path("results/closed_log/issue_152_validation"),
    capture: bool = False,
    expected_preregistration_sha256: str | None = None,
) -> dict[str, object]:
    """Run the frozen #152 validation or fail clearly if capture is unavailable."""
    preregistration = (
        load_preregistration(preregistration_path)
        if expected_preregistration_sha256 is None
        else load_preregistration_snapshot(preregistration_path, expected_preregistration_sha256)
    )
    output.mkdir(parents=True, exist_ok=True)
    traces, manifest, trace_mode = ensure_second_model_traces(output / "traces", preregistration, capture=capture)
    group_size = int(dict(preregistration["quantization"])["group_size"])
    numeric_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    # (pair, candidate, trace, topology+preset) -> one mean cost per module.
    costs: dict[tuple[str, str, str, tuple[object, ...]], float] = {}
    topologies = list(_topologies(preregistration))
    direct_topologies = [topology for topology in topologies if topology.architecture == "direct_dense_histogram"]
    tagged_topologies = [topology for topology in topologies if topology.architecture == "tagged_lane_merge"]
    cost_presets = [str(value) for value in dict(preregistration["bank_cost_grid"])["cost_presets"]]

    for pair in preregistration["frozen_pairs"]:
        members = list(pair["members"])
        specs = [_candidate_spec(pair, member, preregistration) for member in members]
        envelope = AddressEnvelope.from_specs(specs)
        if envelope.phase_count != int(pair["H"]):
            raise RuntimeError("frozen pair does not match its declared H")
        analyses, pair_rows = _evaluate_numeric_pair(
            pair,
            members,
            specs,
            envelope,
            traces,
            group_size,
        )
        numeric_rows.extend(pair_rows)
        for member, spec in zip(members, specs, strict=True):
            candidate = _candidate_name(member)
            for trace in traces:
                heldout_analysis = analyses[(candidate, trace.trace_id)]
                # Direct routing varies with lane width and ports.  The tag
                # table does not: it is dot-scoped and sees the same ordered
                # stream for a fixed (B,map,order).  Cache it once instead of
                # silently shrinking the preregistered grid for run time.
                direct_raw = _direct_routing_components(
                    heldout_analysis,
                    spec,
                    envelope,
                    direct_topologies,
                )
                grid = dict(preregistration["bank_cost_grid"])
                tag_raw = _tagged_component_cache(
                    heldout_analysis,
                    spec,
                    envelope,
                    tagged_topologies,
                    lane_width=min(int(value) for value in grid["lane_widths"]),
                    write_ports_per_bank=min(int(value) for value in grid["write_ports_per_bank"]),
                )
                for topology in direct_topologies:
                    raw = direct_raw[_direct_topology_key(topology)]
                    rows, scored = _scored_cost_rows(
                        pair["id"],
                        candidate,
                        trace,
                        topology,
                        raw.summary,
                        raw.per_dot,
                        cost_presets,
                    )
                    cost_rows.extend(rows)
                    costs.update(scored)
                for topology in tagged_topologies:
                    routing = direct_raw[_direct_topology_key(topology)]
                    tag = tag_raw[_tag_cache_key(topology)]
                    components = [
                        _merge_tagged_components(routing_component, tag_component)
                        for routing_component, tag_component in zip(routing.per_dot, tag.per_dot, strict=True)
                    ]
                    base_summary = dict(routing.summary)
                    base_summary["tag_bits"] = tag.summary["tag_bits"]
                    rows, scored = _scored_cost_rows(
                        pair["id"],
                        candidate,
                        trace,
                        topology,
                        base_summary,
                        components,
                        cost_presets,
                    )
                    cost_rows.extend(rows)
                    costs.update(scored)

    pair_numeric_rows = []
    pair_cost_rows: list[dict[str, object]] = []
    pair_decisions: list[dict[str, object]] = []
    for pair in preregistration["frozen_pairs"]:
        numeric, configuration_rows, decision = _pair_decision(
            pair,
            numeric_rows,
            costs,
            topologies,
            cost_presets,
            traces,
        )
        pair_numeric_rows.append(numeric)
        pair_cost_rows.extend(configuration_rows)
        pair_decisions.append(decision)
    summary = _validation_summary(
        preregistration_path,
        output,
        trace_mode,
        manifest,
        traces,
        pair_decisions,
    )
    _write_validation_artifacts(
        output,
        numeric_rows,
        cost_rows,
        pair_cost_rows,
        pair_numeric_rows,
        pair_decisions,
        summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen #152 held-out radix-cost validation.")
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=Path("results/closed_log/issue_152_validation"))
    parser.add_argument("--capture", action="store_true", help="attempt the required pinned second-model capture if traces are absent")
    parser.add_argument(
        "--expected-preregistration-sha256",
        help="content-pin the frozen preregistration and skip unavailable exploratory source-sweep bytes",
    )
    args = parser.parse_args()
    summary = run_issue152_validation(
        preregistration_path=args.preregistration,
        output=args.output,
        capture=args.capture,
        expected_preregistration_sha256=args.expected_preregistration_sha256,
    )
    print(json.dumps({"decision": summary["decision"], "validated_pair_count": summary["validated_pair_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
