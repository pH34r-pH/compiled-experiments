"""Shared trace analysis and explicitly labelled cost proxies for #146/#149-#155.

None of the operation, storage, or routing figures in this module is a hardware
measurement.  They are transparent reference proxies used to decide whether a
hardware prototype is worth building.  The arithmetic identities, in contrast,
are direct reproductions of CurveFP v2's closed product-address algebra.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np


def _entropy(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if not len(positive):
        return 0.0
    probabilities = positive / positive.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _json_number(value: np.generic | float | int) -> float | int:
    return value.item() if isinstance(value, np.generic) else value


def _keys(phase: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """Structured address keys with stable NumPy equality/unique semantics."""
    keys = np.empty(len(phase), dtype=[("phase", np.int32), ("exponent", np.int32)])
    keys["phase"] = phase.astype(np.int32, copy=False)
    keys["exponent"] = exponent.astype(np.int32, copy=False)
    return keys


def _group_signed_counts(phase: np.ndarray, exponent: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = _keys(phase, exponent)
    unique, inverse, event_counts = np.unique(keys, return_inverse=True, return_counts=True)
    signed = np.bincount(inverse, weights=sign.astype(np.float64), minlength=len(unique))
    return unique, event_counts.astype(np.int64), signed


@lru_cache(maxsize=256)
def _exact_reachable_address_count(E: int, C: int, p: int, q: int, scale_sums: tuple[int, ...]) -> int:
    """Enumerate finite format addresses at the actually observed scale sums.

    This is intentionally not the convenient H×exponent-span envelope: finite
    CurveFP layouts can leave addresses in that rectangle unreachable.
    """
    K = 1 << C
    offset = (1 << (E - 1)) * K
    n = np.arange(1 - offset, (1 << (E + C)) - offset, dtype=np.int64)
    u = p * (n[:, None] + n[None, :])
    denominator = q * K
    divisor = math.gcd(p, denominator)
    phase = np.mod(u, denominator) // divisor
    exponent_base = np.floor_divide(u, denominator)
    fields = []
    for scale_sum in scale_sums:
        key = np.empty(u.size, dtype=[("phase", np.int32), ("exponent", np.int32)])
        key["phase"] = phase.ravel().astype(np.int32)
        key["exponent"] = (exponent_base + scale_sum).ravel().astype(np.int32)
        fields.append(key)
    return int(len(np.unique(np.concatenate(fields)))) if fields else 0


def _reuse_distances(keys: np.ndarray) -> tuple[float, float]:
    """Mean reuse distance and immediate-repeat rate within a lane stream."""
    if len(keys) < 2:
        return float("nan"), 0.0
    previous: dict[tuple[int, int], int] = {}
    distances = []
    for index, key in enumerate(keys):
        name = (int(key["phase"]), int(key["exponent"]))
        if name in previous:
            distances.append(index - previous[name])
        previous[name] = index
    repeats = np.mean(keys[1:] == keys[:-1])
    return (float(np.mean(distances)) if distances else float("nan"), float(repeats))


@dataclass
class TraceAnalysis:
    """Numerical and address statistics for one quantized OperandTrace."""

    summary: dict[str, object]
    per_dot: list[dict[str, object]]
    phase_rows: list[dict[str, object]]
    state_rows: list[dict[str, object]]
    exponent_rows: list[dict[str, object]]
    descriptor_phase: np.ndarray
    descriptor_exponent: np.ndarray
    descriptor_sign: np.ndarray
    descriptor_nonzero: np.ndarray


@dataclass(frozen=True)
class AddressEnvelope:
    """Shared address-space contract for an H-matched radix comparison.

    The native product-exponent domain can differ even when two layouts share
    ``H``.  Charging each member's smaller domain as if it were a workload win
    would confound radix geometry with product-address occupancy.  #152 uses
    this common finite envelope for every compared pair's dense-array and tag
    widths; the native domains are reported separately but not credited in the
    decision statistic.
    """

    phase_count: int
    exponent_min: int
    exponent_max: int

    @property
    def exponent_span(self) -> int:
        return self.exponent_max - self.exponent_min + 1

    @property
    def address_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.phase_count * self.exponent_span)))

    @classmethod
    def from_specs(cls, specs: Sequence[object]) -> "AddressEnvelope":
        """Construct a format-bound common envelope, independent of traces."""
        if not specs:
            raise ValueError("at least one closed-log spec is required")
        phase_count = int(specs[0].H)
        low: list[int] = []
        high: list[int] = []
        for spec in specs:
            if int(spec.H) != phase_count:
                raise ValueError("a shared envelope requires identical H")
            scale_min = -(1 << (int(spec.scale_bits) - 1))
            scale_max = (1 << (int(spec.scale_bits) - 1)) - 1
            denominator = int(spec.q) * int(spec.K)
            low.append(2 * scale_min + math.floor(2 * int(spec.p) * int(spec.n_min) / denominator))
            high.append(2 * scale_max + math.floor(2 * int(spec.p) * int(spec.n_max) / denominator))
        return cls(phase_count=phase_count, exponent_min=min(low), exponent_max=max(high))

    def metadata(self) -> dict[str, int]:
        return {
            "phase_count": self.phase_count,
            "envelope_exponent_min": self.exponent_min,
            "envelope_exponent_max": self.exponent_max,
            "envelope_exponent_span": self.exponent_span,
            "envelope_address_bits": self.address_bits,
        }


@dataclass(frozen=True)
class BankTopology:
    """Predeclared multi-bank placement variation for #152 cost proxies.

    These fields specify a routing experiment, not a CurveFP semantic change.
    Bank count, port count, mapping, and lane order are frozen for every radix
    candidate before looking at held-out traces.
    """

    lane_width: int
    bank_count: int
    write_ports_per_bank: int
    bank_map: str
    lane_order: str = "identity"
    architecture: str = "direct_dense_histogram"
    sets_per_bank: int = 4
    ways_per_set: int = 2

    def __post_init__(self) -> None:
        if self.lane_width < 1:
            raise ValueError("lane_width must be positive")
        if self.bank_count < 1 or self.bank_count & (self.bank_count - 1):
            raise ValueError("bank_count must be a positive power of two")
        if self.write_ports_per_bank < 1:
            raise ValueError("write_ports_per_bank must be positive")
        if self.bank_map not in {"phase_mod", "phase_plus_exponent", "phase_xor_exponent"}:
            raise ValueError("unknown bank_map")
        if self.lane_order not in {"identity", "even_odd_deinterleave", "bit_reverse"}:
            raise ValueError("unknown lane_order")
        if self.architecture not in {"direct_dense_histogram", "tagged_lane_merge"}:
            raise ValueError("unknown architecture")
        if self.sets_per_bank < 1 or self.sets_per_bank & (self.sets_per_bank - 1):
            raise ValueError("sets_per_bank must be a positive power of two")
        if self.ways_per_set < 1:
            raise ValueError("ways_per_set must be positive")


@dataclass
class BankCostAnalysis:
    """Trace-derived cost-proxy outputs, including dot-paired observations."""

    summary: dict[str, object]
    per_dot: list[dict[str, object]]


_COST_PRESETS: dict[str, dict[str, float]] = {
    # Counter updates are the common unit.  The two other presets deliberately
    # sweep routing and tag prices instead of selecting one convenient value.
    "traffic_only": {
        "address_bit": 0.0,
        "replay": 0.0,
        "idle_slot": 0.0,
        "final_read": 0.25,
        "storage_bit": 0.0,
        "tag_compare_bit": 0.0,
        "tag_local_update": 0.0,
        "tag_allocate": 0.0,
    },
    "balanced_simple_ops": {
        "address_bit": 1.0 / 16.0,
        "replay": 1.0,
        "idle_slot": 0.25,
        "final_read": 1.0,
        "storage_bit": 1.0 / 2048.0,
        "tag_compare_bit": 1.0 / 16.0,
        "tag_local_update": 0.25,
        "tag_allocate": 0.25,
    },
    "tag_expensive": {
        "address_bit": 1.0 / 16.0,
        "replay": 4.0,
        "idle_slot": 1.0,
        "final_read": 4.0,
        "storage_bit": 1.0 / 1024.0,
        "tag_compare_bit": 1.0 / 4.0,
        "tag_local_update": 1.0,
        "tag_allocate": 1.0,
    },
}


def _routing_lane_indices(width: int, order: str) -> np.ndarray:
    """Fixed full-dot lane assignments, applied before vector reblocking."""
    indices = np.arange(width, dtype=np.int64)
    if order == "identity":
        return indices
    if order == "even_odd_deinterleave":
        return np.concatenate((indices[::2], indices[1::2]))
    if order == "bit_reverse":
        if width < 1 or width & (width - 1):
            raise ValueError("bit_reverse lane order requires a power-of-two dot width")
        bits = int(math.log2(width))
        values = np.arange(width, dtype=np.uint64)
        reversed_values = np.zeros(width, dtype=np.uint64)
        for _ in range(bits):
            reversed_values = (reversed_values << 1) | (values & 1)
            values >>= 1
        return reversed_values.astype(np.int64)
    raise ValueError(f"unknown lane order: {order}")


def _bank_indices(phase: np.ndarray, exponent: np.ndarray, *, topology: BankTopology, envelope: AddressEnvelope) -> np.ndarray:
    """Apply one fixed power-of-two bank map; no candidate-tuned hashing."""
    mask = topology.bank_count - 1
    offset = exponent.astype(np.int64) - envelope.exponent_min
    if np.any(offset < 0) or np.any(offset >= envelope.exponent_span):
        raise ValueError("observed exponent falls outside the shared address envelope")
    if topology.bank_map == "phase_mod":
        value = phase.astype(np.int64)
    elif topology.bank_map == "phase_plus_exponent":
        value = phase.astype(np.int64) + offset
    else:
        value = phase.astype(np.int64) ^ offset
    return (value & mask).astype(np.int32)


def _constant_multiply_bit_proxy(spec: object) -> int:
    """Shift/add proxy for the non-free ``p * (n_x+n_w)`` address step."""
    p = int(spec.p)
    # A single one-bit coefficient is a wire/shift.  Other set bits require an
    # explicit add in the usual constant-multiply decomposition.
    return max(0, p.bit_count() - 1)


def _native_product_exponent_bounds(spec: object) -> tuple[int, int]:
    scale_min = -(1 << (int(spec.scale_bits) - 1))
    scale_max = (1 << (int(spec.scale_bits) - 1)) - 1
    denominator = int(spec.q) * int(spec.K)
    return (
        2 * scale_min + math.floor(2 * int(spec.p) * int(spec.n_min) / denominator),
        2 * scale_max + math.floor(2 * int(spec.p) * int(spec.n_max) / denominator),
    )


def _routing_group_metrics(
    phase: np.ndarray,
    exponent: np.ndarray,
    valid: np.ndarray,
    *,
    topology: BankTopology,
    envelope: AddressEnvelope,
) -> tuple[int, int, int, int, int]:
    """Return events, service cycles, ideal cycles, replays, and port slots."""
    banks = _bank_indices(phase, exponent, topology=topology, envelope=envelope)
    lane_width = topology.lane_width
    groups = math.ceil(len(banks) / lane_width)
    padded = groups * lane_width
    bank_padded = np.zeros(padded, dtype=np.int32)
    valid_padded = np.zeros(padded, dtype=bool)
    bank_padded[: len(banks)] = banks
    valid_padded[: len(banks)] = valid
    bank_matrix = bank_padded.reshape(groups, lane_width)
    valid_matrix = valid_padded.reshape(groups, lane_width)
    # B is intentionally small and fixed by the preregistration.  This
    # vectorized one-hot count avoids a Python loop over lane groups while
    # preserving exact per-bank contention semantics.
    counts = np.stack(
        [np.sum(valid_matrix & (bank_matrix == bank), axis=1) for bank in range(topology.bank_count)], axis=1
    ).astype(np.int64)
    active = counts.sum(axis=1)
    serviced = np.ceil(counts / topology.write_ports_per_bank).astype(np.int64)
    return (
        int(active.sum()),
        int(serviced.max(axis=1).sum()),
        int(np.ceil(active / (topology.bank_count * topology.write_ports_per_bank)).sum()),
        int(np.maximum(counts - topology.write_ports_per_bank, 0).sum()),
        int((serviced * topology.write_ports_per_bank).sum()),
    )


def _nonzero_signed_bin_count(phase: np.ndarray, exponent: np.ndarray, sign: np.ndarray, valid: np.ndarray) -> int:
    if not np.any(valid):
        return 0
    _, _, signed = _group_signed_counts(phase[valid], exponent[valid], sign[valid])
    return int(np.count_nonzero(signed))


def _tagged_dot_metrics(
    phase: np.ndarray,
    exponent: np.ndarray,
    sign: np.ndarray,
    valid: np.ndarray,
    *,
    topology: BankTopology,
    envelope: AddressEnvelope,
) -> tuple[int, int, int, int, int]:
    """Dot-scoped set-associative write-combining proxy with charged tags."""
    banks = _bank_indices(phase, exponent, topology=topology, envelope=envelope)
    tables: list[list[collections.OrderedDict[tuple[int, int], int]]] = [
        [collections.OrderedDict() for _ in range(topology.sets_per_bank)] for _ in range(topology.bank_count)
    ]
    local_updates = tag_allocations = evictions = main_flush_writes = 0
    tag_compare_bits = 0
    bank_bits = int(math.log2(topology.bank_count))
    set_bits = int(math.log2(topology.sets_per_bank))
    tag_bits = max(1, envelope.address_bits - bank_bits - set_bits)
    for h, t, s, present, bank in zip(phase, exponent, sign, valid, banks, strict=True):
        if not present:
            continue
        code = int(h) * envelope.exponent_span + (int(t) - envelope.exponent_min)
        set_index = (code >> bank_bits) & (topology.sets_per_bank - 1)
        table = tables[int(bank)][set_index]
        key = (int(h), int(t))
        tag_compare_bits += topology.ways_per_set * tag_bits
        local_updates += 1
        if key in table:
            table[key] += int(s)
            table.move_to_end(key)
            continue
        tag_allocations += 1
        if len(table) >= topology.ways_per_set:
            _, prior = table.popitem(last=False)
            evictions += 1
            main_flush_writes += int(prior != 0)
        table[key] = int(s)
    for bank_tables in tables:
        for table in bank_tables:
            main_flush_writes += sum(int(value != 0) for value in table.values())
    return local_updates, tag_compare_bits, tag_allocations, evictions, main_flush_writes


def routing_cost_proxy_from_components(
    components: Mapping[str, int | float], *, architecture: str, cost_preset: str
) -> float:
    """Score already-measured proxy components under one fixed sensitivity preset.

    This small pure helper keeps price sensitivity separate from trace traversal:
    a runner may reuse identical raw traffic/tag observations for all three
    predeclared price presets without accidentally changing the workload.
    """
    if cost_preset not in _COST_PRESETS:
        raise ValueError(f"unknown cost_preset: {cost_preset}")
    weights = _COST_PRESETS[cost_preset]
    events = float(components["active_events"])
    address_bit_ops = float(components["address_generation_bit_ops_proxy"])
    replays = float(components["bank_replays"])
    idle_slots = float(components["idle_bank_port_slots"])
    storage_bits = float(components["storage_bits_proxy"])
    if architecture == "direct_dense_histogram":
        finals = float(components["final_nonzero_signed_bins"])
        return float(
            events
            + weights["address_bit"] * address_bit_ops
            + weights["replay"] * replays
            + weights["idle_slot"] * idle_slots
            + weights["final_read"] * finals
            + weights["storage_bit"] * storage_bits
        )
    if architecture == "tagged_lane_merge":
        flushes = float(components["main_histogram_flush_writes"])
        return float(
            flushes
            + weights["address_bit"] * address_bit_ops
            + weights["replay"] * replays
            + weights["idle_slot"] * idle_slots
            + weights["final_read"] * flushes
            + weights["storage_bit"] * storage_bits
            + weights["tag_compare_bit"] * float(components["tag_compare_bit_ops"])
            + weights["tag_local_update"] * float(components["tag_local_updates"])
            + weights["tag_allocate"] * float(components["tag_allocations"])
        )
    raise ValueError(f"unknown architecture: {architecture}")


def _routing_descriptor_arrays(
    analysis: TraceAnalysis,
    topology: BankTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and lane-order the descriptor arrays used by routing analysis."""
    phase = np.asarray(analysis.descriptor_phase, dtype=np.int32)
    exponent = np.asarray(analysis.descriptor_exponent, dtype=np.int32)
    sign = np.asarray(analysis.descriptor_sign, dtype=np.int8)
    valid = np.asarray(analysis.descriptor_nonzero, dtype=bool)
    if phase.ndim != 2 or phase.shape != exponent.shape or phase.shape != sign.shape or phase.shape != valid.shape:
        raise ValueError("analysis does not carry aligned [dot,width] descriptors")
    order = _routing_lane_indices(phase.shape[1], topology.lane_order)
    return phase[:, order], exponent[:, order], sign[:, order], valid[:, order]


def banked_routing_tag_cost(
    analysis: TraceAnalysis,
    spec: object,
    topology: BankTopology,
    envelope: AddressEnvelope,
    *,
    cost_preset: str = "balanced_simple_ops",
) -> BankCostAnalysis:
    """Explicit multi-bank routing/write/tag proxy for frozen #152 pairs.

    CurveFP already supplies the address and signed-count semantics.  This
    function deliberately measures only engineering choices that CurveFP leaves
    open: vector lane assignment, bank map/ports, direct counter traffic, and a
    costed set-associative write-combiner.  Results are operation/storage/
    traffic proxies, not PPA measurements.
    """
    if cost_preset not in _COST_PRESETS:
        raise ValueError(f"unknown cost_preset: {cost_preset}")
    if int(spec.H) != envelope.phase_count:
        raise ValueError("spec H and address envelope differ")
    phase, exponent, sign, valid = _routing_descriptor_arrays(analysis, topology)
    weights = _COST_PRESETS[cost_preset]
    counter_bits = 1 + math.ceil(math.log2(phase.shape[1] + 1))
    dense_storage_bits = envelope.phase_count * envelope.exponent_span * counter_bits
    bank_bits = int(math.log2(topology.bank_count))
    set_bits = int(math.log2(topology.sets_per_bank))
    tag_bits = max(1, envelope.address_bits - bank_bits - set_bits)
    tag_storage_bits = (
        topology.bank_count
        * topology.sets_per_bank
        * topology.ways_per_set
        * (tag_bits + counter_bits + 2)
        + topology.bank_count * topology.sets_per_bank * max(1, int(math.ceil(math.log2(topology.ways_per_set))))
    )
    phase_decode_bits = max(1, math.ceil(math.log2(int(spec.q) * int(spec.K)))) + max(1, math.ceil(math.log2(int(spec.H))))
    address_bit_ops_per_event = 1 + _constant_multiply_bit_proxy(spec) + phase_decode_bits
    native_min, native_max = _native_product_exponent_bounds(spec)
    per_dot: list[dict[str, object]] = []
    totals: collections.Counter[str] = collections.Counter()

    for dot in range(phase.shape[0]):
        events, cycles, ideal_cycles, replays, port_slots = _routing_group_metrics(
            phase[dot], exponent[dot], valid[dot], topology=topology, envelope=envelope
        )
        final_reads = _nonzero_signed_bin_count(phase[dot], exponent[dot], sign[dot], valid[dot])
        idle_slots = port_slots - events
        address_bit_ops = events * address_bit_ops_per_event
        if topology.architecture == "direct_dense_histogram":
            tagged = {"tag_compare_bit_ops": 0, "tag_allocations": 0, "tag_evictions": 0, "tag_local_updates": 0}
            main_flush_writes = final_reads
            storage_bits = dense_storage_bits
        else:
            local_updates, tag_compare_bits, tag_allocations, tag_evictions, main_flush_writes = _tagged_dot_metrics(
                phase[dot], exponent[dot], sign[dot], valid[dot], topology=topology, envelope=envelope
            )
            tagged = {
                "tag_compare_bit_ops": tag_compare_bits,
                "tag_allocations": tag_allocations,
                "tag_evictions": tag_evictions,
                "tag_local_updates": local_updates,
            }
            storage_bits = tag_storage_bits
        row: dict[str, object] = {
            "dot": dot,
            "active_events": events,
            "routing_cycles": cycles,
            "ideal_routing_cycles": ideal_cycles,
            "bank_conflict_extra_cycles": cycles - ideal_cycles,
            "bank_replays": replays,
            "bank_port_slots": port_slots,
            "idle_bank_port_slots": idle_slots,
            "counter_writes": events if topology.architecture == "direct_dense_histogram" else main_flush_writes,
            "main_histogram_flush_writes": main_flush_writes,
            "final_nonzero_signed_bins": final_reads,
            "address_generation_bit_ops_proxy": address_bit_ops,
            "constant_multiply_shift_add_ops_per_event": _constant_multiply_bit_proxy(spec),
            "phase_decode_bits_per_event": phase_decode_bits,
            "storage_bits_proxy": storage_bits,
            **tagged,
        }
        # Keep the precomputed value and the reusable scorer in lockstep.  The
        # runner reuses these raw components across sensitivity presets.
        row["total_cost_proxy"] = routing_cost_proxy_from_components(
            row, architecture=topology.architecture, cost_preset=cost_preset
        )
        per_dot.append(row)
        for key, value in row.items():
            if key != "dot" and isinstance(value, (int, float, np.integer, np.floating)):
                totals[key] += float(value)
    dots = max(1, len(per_dot))
    summary: dict[str, object] = {
        "classification": "engineering_variation_curvefp_bank_routing_tag_cost_proxy",
        "cost_note": "Explicit operation/storage/traffic proxies; not physical PPA or a new CurveFP accumulator mechanism.",
        "architecture": topology.architecture,
        "lane_width": topology.lane_width,
        "bank_count": topology.bank_count,
        "write_ports_per_bank": topology.write_ports_per_bank,
        "bank_map": topology.bank_map,
        "lane_order": topology.lane_order,
        "sets_per_bank": topology.sets_per_bank,
        "ways_per_set": topology.ways_per_set,
        "cost_preset": cost_preset,
        "p": int(spec.p),
        "q": int(spec.q),
        "native_product_exponent_min": native_min,
        "native_product_exponent_max": native_max,
        "native_product_exponent_span": native_max - native_min + 1,
        "counter_width_bits": counter_bits,
        "tag_bits": tag_bits if topology.architecture == "tagged_lane_merge" else 0,
        **envelope.metadata(),
        "dot_count": len(per_dot),
    }
    for key, value in totals.items():
        summary[f"total_{key}"] = float(value)
        summary[f"mean_{key}_per_dot"] = float(value / dots)
    return BankCostAnalysis(summary=summary, per_dot=per_dot)


def _descriptor_arrays(
    descriptors: object,
    phase_count: int,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = np.asarray(descriptors.phase, dtype=np.int32)
    exponent = np.asarray(descriptors.exponent, dtype=np.int32)
    sign = np.asarray(descriptors.sign, dtype=np.int8)
    nonzero = np.asarray(descriptors.nonzero, dtype=bool)
    if phase.shape != exponent.shape or phase.shape != sign.shape or phase.shape != nonzero.shape or phase.ndim != 2:
        raise ValueError("product descriptors must expose matching [dot,width] arrays")
    if phase_count <= 0:
        raise ValueError("finite phase_count is required for CurveFP address analysis")
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (phase.shape[0],):
        raise ValueError("reference must contain one value per descriptor row")
    return phase, exponent, sign, nonzero, reference


def _global_state_rows(
    all_phase: list[np.ndarray],
    all_exponents: list[np.ndarray],
    all_values: list[np.ndarray],
) -> tuple[float, np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    exponent_values = np.concatenate(all_exponents) if all_exponents else np.array([], dtype=np.int32)
    if not all_phase:
        return 0.0, exponent_values, [], []
    global_keys = _keys(np.concatenate(all_phase), np.concatenate(all_exponents))
    unique_global, inverse_global, event_counts_global = np.unique(
        global_keys, return_inverse=True, return_counts=True,
    )
    values_global = np.concatenate(all_values)
    signed_global = np.bincount(inverse_global, weights=values_global, minlength=len(unique_global))
    absolute_global = np.bincount(inverse_global, weights=np.abs(values_global), minlength=len(unique_global))
    entropy = _entropy(event_counts_global)
    total_absolute = max(float(absolute_global.sum()), 1e-30)
    total_events = max(int(event_counts_global.sum()), 1)
    state_rows = [
        {
            "phase": int(key["phase"]),
            "exponent": int(key["exponent"]),
            "event_count": int(events),
            "event_fraction": float(events / total_events),
            "absolute_contribution": float(absolute),
            "absolute_contribution_fraction": float(absolute / total_absolute),
            "signed_contribution_dataset_sum": float(signed),
            "note": "Dataset aggregate only; it is not cross-dot accumulator cancellation.",
        }
        for key, events, absolute, signed in zip(
            unique_global, event_counts_global, absolute_global, signed_global, strict=True,
        )
    ]
    values, counts = np.unique(exponent_values, return_counts=True)
    exponent_rows = [
        {
            "exponent": int(value),
            "event_count": int(count),
            "event_fraction": float(count / max(1, counts.sum())),
        }
        for value, count in zip(values, counts, strict=True)
    ]
    return entropy, exponent_values, state_rows, exponent_rows


def _phase_rows(
    phase_events: np.ndarray,
    phase_abs: np.ndarray,
    phase_signed: np.ndarray,
    phase_dot_abs: list[np.ndarray],
    phase_dot_event: list[np.ndarray],
) -> list[dict[str, object]]:
    phase_count = len(phase_events)
    phase_abs_matrix = np.asarray(phase_dot_abs) if phase_dot_abs else np.zeros((0, phase_count))
    phase_event_matrix = np.asarray(phase_dot_event) if phase_dot_event else np.zeros((0, phase_count))
    rows: list[dict[str, object]] = []
    for h in range(phase_count):
        event_series = phase_event_matrix[:, h] if len(phase_event_matrix) else np.array([])
        abs_series = phase_abs_matrix[:, h] if len(phase_abs_matrix) else np.array([])
        rows.append({
            "phase": h,
            "event_count": int(phase_events[h]),
            "event_fraction": float(phase_events[h] / max(1, phase_events.sum())),
            "absolute_contribution": float(phase_abs[h]),
            "absolute_contribution_fraction": float(phase_abs[h] / max(float(phase_abs.sum()), 1e-30)),
            "signed_contribution": float(phase_signed[h]),
            "event_std_across_dots": float(np.std(event_series)) if len(event_series) else 0.0,
            "absolute_contribution_std_across_dots": float(np.std(abs_series)) if len(abs_series) else 0.0,
            "active_dots_fraction": float(np.mean(event_series > 0)) if len(event_series) else 0.0,
        })
    return rows


def _phase_kulisch_value(
    phase: np.ndarray,
    exponent: np.ndarray,
    sign: np.ndarray,
    *,
    phase_count: int,
    min_exponent: int,
) -> float:
    """Reconstruct one dot product through exact per-phase integer shifts."""
    phase_integer = np.zeros(phase_count, dtype=object)
    for h in range(phase_count):
        mask = phase == h
        if np.any(mask):
            phase_integer[h] = sum(
                int(delta) << int(value - min_exponent)
                for delta, value in zip(sign[mask], exponent[mask], strict=True)
            )
    return float(
        sum(
            float(phase_integer[h]) * math.exp2(min_exponent + h / phase_count)
            for h in range(phase_count)
        )
    )


def _temporal_locality(
    phase: np.ndarray,
    exponent: np.ndarray,
    *,
    enabled: bool,
) -> tuple[float, float]:
    if not enabled:
        return float("nan"), float("nan")
    return _reuse_distances(_keys(phase, exponent))


def _reachable_addresses(
    descriptors: object,
    dot: int,
    valid: np.ndarray,
    *,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    scale_sums = tuple(
        sorted(
            {
                int(item)
                for item in (
                    descriptors.lhs_scale_exponent[dot, valid]
                    + descriptors.rhs_scale_exponent[dot, valid]
                )
            }
        )
    )
    return _exact_reachable_address_count(
        descriptors.spec.E,
        descriptors.spec.C,
        descriptors.spec.p,
        descriptors.spec.q,
        scale_sums,
    )


def analyze_product_descriptors(
    descriptors: object,
    *,
    phase_count: int,
    reference: np.ndarray,
    context: Mapping[str, object] | None = None,
    validate_kulisch: bool = True,
    compute_exact_reachability: bool = True,
    compute_temporal_locality: bool = True,
) -> TraceAnalysis:
    """Analyze CurveFP-style descriptors without treating its accumulator as new.

    ``descriptors`` is deliberately duck-typed to keep trace analysis separate
    from the datatype implementation.  Exact reachability needs [dot,width]
    ``phase``, ``exponent``, ``sign``, ``nonzero``, lhs/rhs scale-exponent
    arrays, and a finite rational ``spec``; the first four also drive the
    accumulation measurements.  ``validate_kulisch=False`` is an explicit
    address-only screen for a large radix sweep: #146 has already validated
    the exact Kulisch reference, and repeating O(H) reductions for every
    candidate would not add information about address occupancy.  Likewise,
    exact finite reachability and temporal reuse are #146 measurements; the
    radial candidate screen may skip their expensive repetition while retaining
    its required occupancy, entropy, phase, and exponent statistics.
    """
    phase, exponent, sign, nonzero, reference = _descriptor_arrays(
        descriptors, phase_count, reference,
    )

    product_value = sign.astype(np.float64) * np.exp2(exponent.astype(np.float64) + phase.astype(np.float64) / phase_count)
    product_value[~nonzero] = 0.0
    immediate = np.sum(product_value, axis=1)
    histogram = np.zeros_like(immediate)
    kulisch = np.zeros_like(immediate)
    per_dot: list[dict[str, object]] = []
    phase_events = np.zeros(phase_count, dtype=np.int64)
    phase_abs = np.zeros(phase_count, dtype=np.float64)
    phase_signed = np.zeros(phase_count, dtype=np.float64)
    phase_dot_abs: list[np.ndarray] = []
    phase_dot_event: list[np.ndarray] = []
    all_exponents: list[np.ndarray] = []
    all_phase: list[np.ndarray] = []
    all_sign: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    total_unique = 0
    total_active = 0
    all_reuse_distances: list[float] = []
    adjacent_repeat_values: list[float] = []

    for dot in range(phase.shape[0]):
        valid = nonzero[dot]
        p = phase[dot, valid]
        e = exponent[dot, valid]
        s = sign[dot, valid]
        value = product_value[dot, valid]
        active = len(p)
        phase_event = np.bincount(p, minlength=phase_count).astype(np.int64) if active else np.zeros(phase_count, dtype=np.int64)
        phase_abs_dot = np.bincount(p, weights=np.abs(value), minlength=phase_count) if active else np.zeros(phase_count)
        phase_signed_dot = np.bincount(p, weights=value, minlength=phase_count) if active else np.zeros(phase_count)
        phase_events += phase_event
        phase_abs += phase_abs_dot
        phase_signed += phase_signed_dot
        phase_dot_abs.append(phase_abs_dot)
        phase_dot_event.append(phase_event)
        if active:
            unique, event_counts, signed_counts = _group_signed_counts(p, e, s)
            coefficient = np.exp2(unique["exponent"].astype(np.float64) + unique["phase"].astype(np.float64) / phase_count)
            histogram[dot] = float(np.sum(signed_counts * coefficient))
            t_min, t_max = int(e.min()), int(e.max())
            if validate_kulisch:
                kulisch[dot] = _phase_kulisch_value(
                    p, e, s, phase_count=phase_count, min_exponent=t_min,
                )
            mean_reuse, adjacent_repeat = _temporal_locality(
                p,
                e,
                enabled=compute_temporal_locality,
            )
            if compute_temporal_locality:
                if math.isfinite(mean_reuse):
                    all_reuse_distances.append(mean_reuse)
                adjacent_repeat_values.append(adjacent_repeat)
            exponent_span = t_max - t_min + 1
            reachable_envelope = phase_count * exponent_span
            reachable_exact = _reachable_addresses(
                descriptors,
                dot,
                valid,
                enabled=compute_exact_reachability,
            )
            unique_states = len(unique)
            entropy = _entropy(event_counts)
            total_unique += unique_states
            total_active += active
            all_exponents.append(e)
            all_phase.append(p)
            all_sign.append(s)
            all_values.append(value)
        else:
            t_min = t_max = 0
            exponent_span = 0
            reachable_envelope = 0
            reachable_exact = 0 if compute_exact_reachability else None
            unique_states = 0
            entropy = 0.0
            mean_reuse, adjacent_repeat = float("nan"), 0.0
        width = phase.shape[1]
        counter_width = 1 + math.ceil(math.log2(width + 1))
        # This is the conservative per-phase bit width of the Kulisch vector:
        # signed D-term sum plus observed exponent displacement.  It is a proxy,
        # not a layout claim for any particular implementation.
        kulisch_width = counter_width + exponent_span
        per_dot.append(
            {
                "dot": dot,
                "width": width,
                "active_products": active,
                "unique_addresses": unique_states,
                "nonzero_signed_histogram_bins": int(np.count_nonzero(signed_counts)) if active else 0,
                "reachable_addresses_exact_observed_scales": reachable_exact,
                "occupied_fraction_exact_observed_scales": (
                    float(unique_states / reachable_exact) if reachable_exact else (0.0 if compute_exact_reachability else float("nan"))
                ),
                "reachable_addresses_upper_envelope": reachable_envelope,
                "occupied_fraction_upper_envelope": float(unique_states / reachable_envelope) if reachable_envelope else 0.0,
                "reuse_factor": float(active / unique_states) if unique_states else 0.0,
                "duplicate_address_rate": float(1 - unique_states / active) if active else 0.0,
                "address_entropy_bits": entropy,
                "min_exponent": t_min,
                "max_exponent": t_max,
                "exponent_span": exponent_span,
                "counter_width_bits": counter_width,
                "kulisch_phase_width_proxy_bits": kulisch_width,
                "immediate_materializations": active,
                "histogram_materializations": int(np.count_nonzero(signed_counts)) if active else 0,
                "kulisch_phase_materializations": int(np.count_nonzero(phase_event)),
                "mean_reuse_distance": mean_reuse,
                "adjacent_address_repeat_rate": adjacent_repeat,
                "immediate_value": float(immediate[dot]),
                "histogram_value": float(histogram[dot]),
                "kulisch_value": float(kulisch[dot]) if validate_kulisch else float("nan"),
            }
        )

    quant_error = immediate - reference
    denom = max(float(np.sum(reference**2)), 1e-30)
    hist_gap = float(np.max(np.abs(histogram - immediate))) if len(immediate) else 0.0
    kulisch_gap = float(np.max(np.abs(kulisch - immediate))) if validate_kulisch and len(immediate) else float("nan")
    global_address_entropy, exponent_values, state_rows, exponent_rows = _global_state_rows(
        all_phase, all_exponents, all_values,
    )
    phase_rows = _phase_rows(
        phase_events, phase_abs, phase_signed, phase_dot_abs, phase_dot_event,
    )
    summary: dict[str, object] = {
        "dot_count": int(phase.shape[0]),
        "width": int(phase.shape[1]),
        "phase_count": phase_count,
        "active_products": int(total_active),
        "mean_unique_addresses_per_dot": float(np.mean([row["unique_addresses"] for row in per_dot])) if per_dot else 0.0,
        "mean_occupied_fraction_exact_observed_scales": (
            float(np.mean([row["occupied_fraction_exact_observed_scales"] for row in per_dot]))
            if per_dot and compute_exact_reachability
            else float("nan")
        ),
        "mean_occupied_fraction_upper_envelope": float(np.mean([row["occupied_fraction_upper_envelope"] for row in per_dot])) if per_dot else 0.0,
        "mean_reuse_factor": float(np.mean([row["reuse_factor"] for row in per_dot])) if per_dot else 0.0,
        "aggregate_duplicate_address_rate": float(1 - total_unique / total_active) if total_active else 0.0,
        "address_entropy_bits": global_address_entropy,
        "product_exponent_min": int(exponent_values.min()) if len(exponent_values) else 0,
        "product_exponent_max": int(exponent_values.max()) if len(exponent_values) else 0,
        "product_exponent_span": int(exponent_values.max() - exponent_values.min() + 1) if len(exponent_values) else 0,
        "product_exponent_entropy_bits": _entropy(np.asarray([row["event_count"] for row in exponent_rows], dtype=np.int64)),
        "phase_event_entropy_bits": _entropy(phase_events),
        "phase_abs_contribution_entropy_bits": _entropy(phase_abs),
        "mean_reuse_distance": float(np.mean(all_reuse_distances)) if all_reuse_distances else float("nan"),
        "adjacent_address_repeat_rate": float(np.mean(adjacent_repeat_values)) if adjacent_repeat_values else 0.0,
        "quantized_dot_nmse": float(np.sum(quant_error**2) / denom),
        "quantized_dot_mae": float(np.mean(np.abs(quant_error))),
        "quantized_dot_bias": float(np.mean(quant_error)),
        "histogram_vs_immediate_max_abs_gap": hist_gap,
        "kulisch_vs_immediate_max_abs_gap": kulisch_gap,
        "kulisch_validation_executed": validate_kulisch,
        "exact_reachability_enumerated": compute_exact_reachability,
        "temporal_locality_enumerated": compute_temporal_locality,
        "visited_product_addresses": total_unique,
        "nonzero_signed_histogram_bins_per_dot_total": int(sum(int(row["nonzero_signed_histogram_bins"]) for row in per_dot)),
        "histogram_materialization_reduction": float(1 - sum(int(row["histogram_materializations"]) for row in per_dot) / total_active) if total_active else 0.0,
        "curvefp_classification": "reproduction_validation",
        "cost_note": "Operation/storage figures are transparent proxies, not hardware measurements.",
    }
    if context:
        summary.update({str(key): value for key, value in context.items()})
    return TraceAnalysis(summary, per_dot, phase_rows, state_rows, exponent_rows, phase, exponent, sign, nonzero)


def lane_coalescing_analysis(analysis: TraceAnalysis, lane_widths: Iterable[int] = (8, 16, 32, 64)) -> list[dict[str, object]]:
    """#150's pre-routing merge measurement and a transparent break-even proxy."""
    phase, exponent, nonzero = analysis.descriptor_phase, analysis.descriptor_exponent, analysis.descriptor_nonzero
    rows: list[dict[str, object]] = []
    for lanes in lane_widths:
        groups = 0
        active_total = unique_total = nonzero_signed_total = bank_before = bank_after = 0
        comparison_ops = shuffle_ops = popcount_ops = 0
        for dot in range(phase.shape[0]):
            for start in range(0, phase.shape[1], lanes):
                valid = nonzero[dot, start : start + lanes]
                p, e = phase[dot, start : start + lanes][valid], exponent[dot, start : start + lanes][valid]
                if not len(p):
                    continue
                keys = _keys(p, e)
                unique = np.unique(keys)
                _, inverse, _ = np.unique(keys, return_inverse=True, return_counts=True)
                signed_counts = np.bincount(inverse, weights=np.asarray(analysis.descriptor_sign[dot, start : start + lanes][valid], dtype=np.float64), minlength=len(unique))
                nonzero_unique = unique[signed_counts != 0]
                active, distinct, nonzero_distinct = len(keys), len(unique), len(nonzero_unique)
                active_total += active
                unique_total += distinct
                nonzero_signed_total += nonzero_distinct
                groups += 1
                # Phase-indexed banks are a proxy for CurveFP bank pressure. A
                # unique address is still one routed write after coalescing.
                banks_before = len(np.unique(p))
                banks_after = len(np.unique(nonzero_unique["phase"]))
                bank_before += active - banks_before
                bank_after += nonzero_distinct - banks_after
                levels = math.ceil(math.log2(max(2, active)))
                comparison_ops += active * levels
                shuffle_ops += active
                popcount_ops += active - distinct
        avoided = active_total - nonzero_signed_total
        overhead = comparison_ops + shuffle_ops + popcount_ops
        rows.append(
            {
                "lane_count": lanes,
                "groups": groups,
                "active_events": active_total,
                "visited_unique_addresses": unique_total,
                "nonzero_signed_merged_addresses": nonzero_signed_total,
                "address_coalescing_ratio": float(active_total / unique_total) if unique_total else 0.0,
                "effective_write_coalescing_ratio": float(active_total / nonzero_signed_total) if nonzero_signed_total else None,
                "accumulator_writes_avoided": avoided,
                "writes_avoided_fraction": float(avoided / active_total) if active_total else 0.0,
                "bank_conflict_proxy_before": bank_before,
                "bank_conflict_proxy_after": bank_after,
                "bank_conflict_proxy_reduction": float(1 - bank_after / bank_before) if bank_before else 0.0,
                "comparison_ops_proxy": comparison_ops,
                "shuffle_ops_proxy": shuffle_ops,
                "popcount_add_ops_proxy": popcount_ops,
                "break_even_update_cost_in_simple_op_units": float(overhead / avoided) if avoided else float("inf"),
                "classification": "engineering_variation_of_curvefp_duplicate_merge",
            }
        )
    return rows


def phase_specialization_analysis(
    analysis: TraceAnalysis,
    omission_fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.125, 0.25),
) -> list[dict[str, object]]:
    """#151 frequency/contribution statistics plus controlled omission/merge tests."""
    phase, exponent, sign, valid = (
        analysis.descriptor_phase,
        analysis.descriptor_exponent,
        analysis.descriptor_sign,
        analysis.descriptor_nonzero,
    )
    H = int(analysis.summary["phase_count"])
    values = sign.astype(np.float64) * np.exp2(exponent.astype(np.float64) + phase.astype(np.float64) / H)
    values[~valid] = 0.0
    immediate = values.sum(axis=1)
    frequency = np.bincount(phase[valid], minlength=H) if np.any(valid) else np.zeros(H, dtype=np.int64)
    order = np.argsort(frequency, kind="stable")
    rows = []
    for fraction in omission_fractions:
        target_events = int(math.floor(fraction * frequency.sum()))
        selected: list[int] = []
        count = 0
        for h in order:
            if count + frequency[h] > target_events:
                break
            selected.append(int(h))
            count += int(frequency[h])
        mask = valid & np.isin(phase, np.asarray(selected, dtype=np.int32))
        omitted = values.copy()
        omitted[mask] = 0.0
        omitted_result = omitted.sum(axis=1)
        # Approximate merge uses the nearest kept coefficient on the circular
        # phase ring (including an exponent carry across H→0).  This is *not*
        # CurveFP semantics and is explicitly reported as a new error.
        merged = values.copy()
        kept = [h for h in range(H) if h not in selected]
        if kept:
            for h in selected:
                target, effective_target = min(
                    ((candidate, candidate + wrap * H) for candidate in kept for wrap in (-1, 0, 1)),
                    key=lambda item: abs(item[1] - h),
                )
                current = valid & (phase == h)
                merged[current] *= math.exp2((effective_target - h) / H)
        merged_result = merged.sum(axis=1)
        denom = max(float(np.sum(immediate**2)), 1e-30)
        rows.append(
            {
                "rare_event_fraction_target": fraction,
                "omitted_phases": selected,
                "omitted_event_fraction": float(count / max(1, frequency.sum())),
                "bank_activity_reduction_fraction": float(len(selected) / H),
                "selected_any_phase": bool(selected),
                "omission_nmse_vs_quantized_dot": float(np.sum((omitted_result - immediate) ** 2) / denom),
                "merge_nmse_vs_quantized_dot": float(np.sum((merged_result - immediate) ** 2) / denom),
                "classification": "new_approximate_mechanism_requires_error_reporting",
            }
        )
    return rows


def associative_accumulator_analysis(
    analysis: TraceAnalysis,
    capacities: Iterable[int] = (8, 16, 32, 64, 128),
) -> list[dict[str, object]]:
    """#149 LRU associative accumulator / spill traffic proxy.

    Dense histogram occupancy is an exact CurveFP reference.  The associative
    alternatives are engineering models: they pay tag comparisons and explicit
    spills, and this reference does not claim a microarchitectural result.
    """
    phase, exponent, valid = analysis.descriptor_phase, analysis.descriptor_exponent, analysis.descriptor_nonzero
    H = int(analysis.summary["phase_count"])
    spans = [int(row["exponent_span"]) for row in analysis.per_dot]
    reachable = H * max(spans, default=0)
    tag_bits = max(1, math.ceil(math.log2(max(2, reachable))))
    dense_counter_bits = max(int(row["counter_width_bits"]) for row in analysis.per_dot) if analysis.per_dot else 0
    rows = []
    for capacity in capacities:
        spills = hits = misses = comparisons = final_resident = 0
        for dot in range(phase.shape[0]):
            table: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
            for h, e, present in zip(phase[dot], exponent[dot], valid[dot], strict=True):
                if not present:
                    continue
                key = (int(h), int(e))
                comparisons += min(capacity, max(1, len(table)))
                if key in table:
                    hits += 1
                    table.move_to_end(key)
                else:
                    misses += 1
                    if len(table) >= capacity:
                        table.popitem(last=False)
                        spills += 1
                    table[key] = None
            final_resident += len(table)
        active = int(analysis.summary["active_products"])
        dense_storage = reachable * dense_counter_bits
        associative_storage = capacity * (tag_bits + dense_counter_bits)
        # Dense history has one update per product.  Associative traffic includes
        # eviction write + final merge per resident state, in addition to tagged
        # hit/miss update.  Comparison count is separate and never free.
        dense_updates = active
        associative_traffic = active + spills + final_resident
        rows.append(
            {
                "capacity": capacity,
                "active_events": active,
                "hits": hits,
                "misses": misses,
                "hit_rate": float(hits / active) if active else 0.0,
                "spills": spills,
                "final_resident_entries": final_resident,
                "tag_compare_ops_proxy": comparisons,
                "dense_histogram_counter_storage_bits_proxy": dense_storage,
                "associative_storage_bits_proxy": associative_storage,
                "dense_update_traffic_proxy": dense_updates,
                "associative_update_spill_merge_traffic_proxy": associative_traffic,
                "traffic_delta_vs_dense": associative_traffic - dense_updates,
                "classification": "engineering_alternative_to_curvefp_histogram_and_kulisch",
            }
        )
    return rows


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    values_x, values_y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(values_x) < 2 or np.std(values_x) == 0 or np.std(values_y) == 0:
        return float("nan")
    return float(np.corrcoef(values_x, values_y)[0, 1])


def pareto_front(rows: Iterable[Mapping[str, object]], minimize: Iterable[str]) -> list[dict[str, object]]:
    """Return rows not dominated on every named numeric objective."""
    candidates = [dict(row) for row in rows]
    objectives = tuple(minimize)
    front = []
    for index, candidate in enumerate(candidates):
        values = [float(candidate[name]) for name in objectives]
        if not all(math.isfinite(value) for value in values):
            continue
        dominated = False
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_values = [float(other[name]) for name in objectives]
            if all(math.isfinite(value) for value in other_values) and all(a <= b for a, b in zip(other_values, values)) and any(a < b for a, b in zip(other_values, values)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front
