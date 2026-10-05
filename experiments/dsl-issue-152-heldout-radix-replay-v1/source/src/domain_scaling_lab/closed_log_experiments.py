"""Deterministic experiment runner for the closed-log / CurveFP program.

This module is deliberately a coordinator, not another datatype
implementation.  ``closed_log`` owns codes, block scales, exact product
addresses, and their reconstruction.  ``closed_log_traces`` owns the stable
operand dataset; ``closed_log_analysis`` owns CurveFP-style histogram/Kulisch
analysis and explicitly-labelled engineering cost proxies.  Keeping those
boundaries makes #146's captured operands reusable by #149--#155.

The runner has two purposes:

* generate compact JSON/CSV evidence from a deterministic operand trace set;
* make the distinction between CurveFP reproduction and extensions visible in
  every row's ``classification`` field.

It does not claim hardware results.  All routing, coalescing, and associative
figures are operation/storage/traffic proxies defined in
``closed_log_analysis``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .closed_log import ClosedLogSpec, decode_grouped, product_descriptors, quantize_grouped
from .closed_log_analysis import (
    TraceAnalysis,
    analyze_product_descriptors,
    associative_accumulator_analysis,
    lane_coalescing_analysis,
    pareto_front,
    pearson_r,
    phase_specialization_analysis,
)
from .closed_log_traces import (
    OperandTrace,
    build_default_trace_dataset,
    load_operand_traces,
    save_product_address_trace,
    write_product_address_manifest,
)
from .factor_algebra import synthetic_distributions


# ``closed_log_analysis`` identifies the reference algebra as CurveFP v2.  The
# user-facing memo records the full bibliographic reference; keeping the date
# here makes standalone results auditable as well.
CURVEFP_REFERENCE = "CurveFP arXiv:2608.10010v2 (consulted 2026-09-01)"
DEFAULT_SEED = 146
DEFAULT_GROUP_SIZES = (32, 64, 128, 256, 512)
DEFAULT_LANE_SIZES = (8, 16, 32, 64)
DEFAULT_ASSOCIATIVE_CAPACITIES = (8, 16, 32, 64, 128)


@dataclass(frozen=True)
class GenericLogEncoding:
    """A scalar-grid control, independent from the exact closed-log datatype.

    ``codes`` are nonnegative rank IDs: zero is exact zero and positive ranks
    index the magnitude grid.  The purpose of this small helper is only to
    sweep arbitrary real radices, including exact ``e``.  Exact product-address
    analysis remains restricted to rational-power-of-two ``ClosedLogSpec``
    layouts.
    """

    decoded: np.ndarray
    codes: np.ndarray
    scale_exponent: np.ndarray
    clipped: np.ndarray
    levels: np.ndarray


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _as_json(value: object) -> object:
    """Convert NumPy/path values and non-finite floats to strict JSON values."""
    if isinstance(value, np.generic):
        return _as_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _as_json(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_value(value: object) -> object:
    value = _as_json(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write deterministic heterogeneous rows without silently dropping fields."""
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _nmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    return float(np.sum((reconstructed - reference) ** 2) / max(float(np.sum(reference**2)), 1e-30))


def _element_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    error = np.asarray(reconstructed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "scalar_nmse": _nmse(reference, reconstructed),
        "scalar_mae": float(np.mean(np.abs(error))),
        "tail_abs_error_p99": float(np.quantile(np.abs(error), 0.99)),
        "tail_abs_error_p999": float(np.quantile(np.abs(error), 0.999)),
        "scalar_bias": float(np.mean(error)),
    }


def _normalise_shape(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Expose a last-axis grouping view and remember the source shape."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1), array.shape
    if array.ndim == 1:
        return array.reshape(1, -1), array.shape
    return array.reshape(-1, array.shape[-1]), array.shape


def _restore_shape(values: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    if not original_shape:
        return values.reshape(())
    return values.reshape(original_shape)


def _positive_level_count(E: int, C: int) -> int:
    # One all-zero magnitude rank is reserved for exact zero.
    return (1 << (E + C)) - 1


def _grid_levels(*, E: int, C: int, radix: float) -> np.ndarray:
    """Return a centred CurveFP-like scalar grid in its physical magnitudes.

    The grid is ``radix ** (n / K)`` rather than merely ``radix ** n`` so that
    C remains the finite fractional-coordinate field.  This matches the
    fractional-coordinate form used by the closed rational layout while still
    permitting irrational scalar controls such as exact ``e``.
    """
    if not (radix > 1.0 and math.isfinite(radix)):
        raise ValueError("radix must be finite and greater than one")
    K = 1 << C
    count = _positive_level_count(E, C)
    n = np.arange(count, dtype=np.float64) - (count // 2)
    # This is the one canonical real-valued grid for both ln/log2 controls.
    # A base conversion must never introduce a distinct representational grid.
    levels = np.power(float(radix), n / K)
    return levels.astype(np.float64)


def quantize_generic_log_grouped(
    values: np.ndarray,
    *,
    E: int,
    C: int,
    radix: float,
    group_size: int | None,
    log_kind: str = "log2",
    scale_policy: str = "curvefp_ceil_absmax",
) -> GenericLogEncoding:
    """Quantize to an arbitrary-radix, power-of-two-block-scaled log grid.

    ``log_kind`` exists solely for the explicit ln-vs-log2 equivalence control.
    It is evaluated as a diagnostic coordinate identity but never chooses a
    code: nearest physical-level selection uses one canonical grid.
    """
    if log_kind not in {"log2", "ln"}:
        raise ValueError("log_kind must be 'log2' or 'ln'")
    if scale_policy not in {"curvefp_ceil_absmax", "endpoint_cover"}:
        raise ValueError("scale_policy must be 'curvefp_ceil_absmax' or 'endpoint_cover'")
    rows, original_shape = _normalise_shape(values)
    width = rows.shape[1]
    size = width if group_size is None else int(group_size)
    if size < 1:
        raise ValueError("group_size must be positive or None")
    # Selection is nearest *absolute reconstructed magnitude*, matching the
    # corrected CurveFP scalar reference rather than rounding a log coordinate.
    levels = _grid_levels(E=E, C=C, radix=radix)
    rank = np.zeros_like(rows, dtype=np.int32)
    decoded = np.zeros_like(rows, dtype=np.float32)
    clipped = np.zeros_like(rows, dtype=bool)
    scales: list[np.ndarray] = []
    n_high = len(levels) - 1
    maximum_level_log2 = math.log2(float(levels[-1]))

    for start in range(0, width, size):
        stop = min(start + size, width)
        block = rows[:, start:stop]
        magnitude = np.abs(block).astype(np.float64)
        maximum = np.max(magnitude, axis=1)
        log2_maximum = np.log2(np.maximum(maximum, np.finfo(np.float64).tiny))
        base_scale = np.ceil(log2_maximum)
        if scale_policy == "endpoint_cover":
            # Match ClosedLogSpec exactly: take ceil *after* compensating for
            # the top local level; subtraction after ceil is not equivalent.
            base_scale = np.ceil(log2_maximum - maximum_level_log2)
        exponent = np.where(maximum > 0.0, base_scale, 0.0)
        # Block exponents are execution/metadata state, so do not silently use
        # unrestricted floating scales.
        exponent = np.clip(exponent, -127, 127).astype(np.int16)
        scale = np.exp2(exponent.astype(np.float64))
        normalised = magnitude / scale[:, None]
        nonzero = normalised > 0.0
        # Evaluate the requested base-converted coordinate only as a control.
        # It is deliberately not used for quantization, because both paths
        # represent the same canonical physical levels.
        with np.errstate(divide="ignore", invalid="ignore"):
            if log_kind == "log2":
                _coordinate_diagnostic = np.log2(normalised[nonzero]) / math.log2(radix)
            else:
                _coordinate_diagnostic = np.log(normalised[nonzero]) / math.log(radix)
        del _coordinate_diagnostic
        right = np.clip(np.searchsorted(levels, normalised, side="left"), 0, n_high)
        left = np.maximum(right - 1, 0)
        # Deterministic lower tie-break, the same convention as the retained
        # scalar lattice work.
        position = np.where(
            np.abs(levels[right] - normalised) < np.abs(levels[left] - normalised), right, left
        ).astype(np.int64)
        rank[:, start:stop][nonzero] = position[nonzero].astype(np.int32) + 1
        block_decoded = np.zeros_like(block, dtype=np.float32)
        reconstructed = np.sign(block).astype(np.float64) * scale[:, None] * levels[position]
        block_decoded[nonzero] = reconstructed[nonzero].astype(np.float32)
        decoded[:, start:stop] = block_decoded
        clipped[:, start:stop] = nonzero & ((normalised < levels[0]) | (normalised > levels[-1]))
        scales.append(exponent)

    return GenericLogEncoding(
        decoded=_restore_shape(decoded, original_shape),
        codes=_restore_shape(rank, original_shape),
        scale_exponent=np.stack(scales, axis=1),
        clipped=_restore_shape(clipped, original_shape),
        levels=levels,
    )


def _bf16_round_trip(values: np.ndarray) -> np.ndarray:
    """IEEE round-to-nearest-even BF16 reference using float32 bit patterns."""
    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def block_int8_grouped(values: np.ndarray, group_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Power-of-two block-scaled signed INT8 baseline and its scale exponents."""
    rows, original_shape = _normalise_shape(values)
    width = rows.shape[1]
    size = width if group_size is None else int(group_size)
    if size < 1:
        raise ValueError("group_size must be positive or None")
    decoded = np.zeros_like(rows, dtype=np.float32)
    scales: list[np.ndarray] = []
    for start in range(0, width, size):
        stop = min(start + size, width)
        block = rows[:, start:stop]
        maximum = np.max(np.abs(block).astype(np.float64), axis=1)
        exponent = np.where(maximum > 0.0, np.ceil(np.log2(np.maximum(maximum, np.finfo(np.float64).tiny))), 0.0)
        exponent = np.clip(exponent, -127, 127).astype(np.int16)
        scale = np.exp2(exponent.astype(np.float64))
        code = np.clip(np.rint(block.astype(np.float64) * 127.0 / scale[:, None]), -127, 127)
        decoded[:, start:stop] = (code * scale[:, None] / 127.0).astype(np.float32)
        scales.append(exponent)
    return _restore_shape(decoded, original_shape), np.stack(scales, axis=1)


def _state_entropy(codes: np.ndarray) -> float:
    codes = np.asarray(codes)
    active = codes[codes != 0]
    if not len(active):
        return 0.0
    _, counts = np.unique(active, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _generic_scalar_metrics(
    values: np.ndarray,
    encoding: GenericLogEncoding,
    *,
    E: int,
    C: int,
    radix: float,
    group_size: int | None,
    scale_policy: str,
) -> dict[str, object]:
    active = encoding.codes[encoding.codes != 0]
    count = _positive_level_count(E, C)
    return {
        **_element_metrics(values, encoding.decoded),
        "E": E,
        "C": C,
        "element_bits": 1 + E + C,
        "radix": radix,
        "scale_policy": scale_policy,
        "adjacent_level_ratio": float(radix ** (1.0 / (1 << C))),
        "dynamic_range": float(encoding.levels[-1] / encoding.levels[0]),
        "clipping_rate": float(np.mean(encoding.clipped)),
        "occupied_scalar_states": int(len(np.unique(active))) if len(active) else 0,
        "available_scalar_states": count,
        "occupied_scalar_state_fraction": float(len(np.unique(active)) / count) if len(active) else 0.0,
        "state_entropy_bits": _state_entropy(encoding.codes),
        "scale_metadata_bits_per_value": float(8.0 / (group_size if group_size is not None else np.asarray(values).shape[-1])),
        "group_size": group_size if group_size is not None else "tensorwise",
    }


def _endpoint_hit_rate(payload: object) -> float:
    """Return a clearly-labelled endpoint proxy when exact clipping is absent."""
    sign = np.asarray(getattr(payload, "sign"))
    rank = np.asarray(getattr(payload, "rank"))
    active = sign != 0
    if not np.any(active):
        return 0.0
    values = rank[active]
    return float(np.mean((values == values.min()) | (values == values.max())))


def _core_scalar_metrics(
    values: np.ndarray,
    *,
    spec: ClosedLogSpec,
    group_size: int | None,
) -> tuple[dict[str, object], object, np.ndarray]:
    payload = quantize_grouped(values, spec, group_size=group_size)
    decoded = decode_grouped(payload)
    metadata = dict(spec.metadata())
    result: dict[str, object] = {
        **_element_metrics(values, decoded),
        **metadata,
        "E": spec.E,
        "C": spec.C,
        "p": spec.p,
        "q": spec.q,
        "H": spec.H,
        "K": spec.K,
        "radix": spec.radix,
        "scale_policy": spec.scale_policy,
        "element_bits": spec.total_bits,
        "group_size": group_size if group_size is not None else "tensorwise",
        "scale_metadata_bits_per_value": float(spec.scale_bits / (group_size if group_size is not None else np.asarray(values).shape[-1])),
        "endpoint_hit_rate_proxy": _endpoint_hit_rate(payload),
        "clipping_rate": float(np.mean(np.asarray(payload.clipped))),
        "scale_clipping_rate": float(np.mean(np.asarray(payload.scale_clipped))),
        "clipping_note": "ClosedLogPayload records source coordinates outside the finite local grid.",
    }
    return result, payload, decoded


def _trace_context(trace: OperandTrace) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "trace_width": trace.width,
        "trace_dot_count": trace.dot_count,
        "model": trace.metadata.get("model"),
        "model_revision": trace.metadata.get("revision"),
        "module": trace.metadata.get("module"),
        "layer": trace.metadata.get("layer"),
    }


def _trace_baseline_row(
    trace: OperandTrace,
    *,
    representation: str,
    left: np.ndarray,
    right: np.ndarray,
    group_size: int | None,
    element_bits: int,
    classification: str,
) -> dict[str, object]:
    dot = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    row = {
        **_trace_context(trace),
        "representation": representation,
        "classification": classification,
        "group_size": group_size if group_size is not None else "tensorwise",
        "element_bits": element_bits,
        "scale_metadata_bits_per_operand_value": 0.0 if group_size is None else float(8.0 / group_size),
        "quantized_dot_nmse": _nmse(trace.reference, dot),
        "quantized_dot_mae": float(np.mean(np.abs(dot - trace.reference))),
        "requantization_materializations": int(trace.width * trace.dot_count),
        "address_analysis": "not_applicable_to_nonclosed_log_baseline",
    }
    return row


def ensure_operand_traces(
    trace_dir: Path,
    *,
    capture: bool,
    include_real: bool,
    local_files_only: bool,
    seed: int,
) -> tuple[list[OperandTrace], dict[str, object], str]:
    """Reuse a trace manifest or make an explicitly requested deterministic one.

    A missing manifest is *not* an invitation to implicitly download a model.
    The fallback is the retained synthetic trace set; ``--capture`` is required
    to request fresh pretrained capture.
    """
    manifest_path = trace_dir / "manifest.json"
    if manifest_path.exists() and not capture:
        traces, manifest = load_operand_traces(trace_dir)
        return traces, manifest, "reused"
    build_default_trace_dataset(
        trace_dir,
        include_real=include_real if capture else False,
        local_files_only=local_files_only,
        seed=seed,
    )
    traces, manifest = load_operand_traces(trace_dir)
    return traces, manifest, "captured" if capture else "synthetic_fallback"


def _effective_group_size(trace: OperandTrace, group_size: int | None) -> int | None:
    if group_size is None:
        return None
    return min(int(group_size), trace.width)


def evaluate_closed_log_trace(
    trace: OperandTrace,
    *,
    spec: ClosedLogSpec,
    group_size: int | None,
    label: str,
    classification: str,
    issue: str,
    validate_kulisch: bool = True,
    compute_exact_reachability: bool = True,
    compute_temporal_locality: bool = True,
) -> tuple[TraceAnalysis, dict[str, object], object, object]:
    """Quantize one true GEMM trace and analyze its CurveFP addresses.

    Full histogram/Kulisch reproduction is the default.  Large rational radix
    sweeps may opt into the explicitly marked address-only path after #146 has
    already established numerical agreement of the accumulation alternatives.
    """
    effective_group = _effective_group_size(trace, group_size)
    lhs = quantize_grouped(trace.a, spec, group_size=effective_group)
    rhs = quantize_grouped(trace.b, spec, group_size=effective_group)
    descriptors = product_descriptors(lhs, rhs)
    context: dict[str, object] = {
        **_trace_context(trace),
        "representation": label,
        "classification": classification,
        "issue": issue,
        "curvefp_reference": CURVEFP_REFERENCE,
        "group_size": effective_group if effective_group is not None else "tensorwise",
        "E": spec.E,
        "C": spec.C,
        "p": spec.p,
        "q": spec.q,
        "H": spec.H,
        "K": spec.K,
        "radix": spec.radix,
        "scale_policy": spec.scale_policy,
        "element_bits": spec.total_bits,
        "scale_metadata_bits_per_operand_value": float(
            spec.scale_bits / (effective_group if effective_group is not None else trace.width)
        ),
    }
    analysis = analyze_product_descriptors(
        descriptors,
        phase_count=spec.H,
        reference=trace.reference,
        context=context,
        validate_kulisch=validate_kulisch,
        compute_exact_reachability=compute_exact_reachability,
        compute_temporal_locality=compute_temporal_locality,
    )
    left_decoded = decode_grouped(lhs)
    right_decoded = decode_grouped(rhs)
    decoded_dot = np.sum(left_decoded.astype(np.float64) * right_decoded.astype(np.float64), axis=1)
    row = dict(analysis.summary)
    row.update(
        {
            "lhs_scalar_nmse": _nmse(trace.a, left_decoded),
            "rhs_scalar_nmse": _nmse(trace.b, right_decoded),
            "lhs_clipping_rate": float(np.mean(lhs.clipped)),
            "rhs_clipping_rate": float(np.mean(rhs.clipped)),
            "lhs_scale_clipping_rate": float(np.mean(lhs.scale_clipped)),
            "rhs_scale_clipping_rate": float(np.mean(rhs.scale_clipped)),
            "lhs_scale_metadata_entries": int(np.asarray(lhs.scale_exponent).size),
            "rhs_scale_metadata_entries": int(np.asarray(rhs.scale_exponent).size),
            "input_code_conversion_proxy": int(trace.a.size + trace.b.size),
            "scale_selection_address_proxy": int(np.asarray(lhs.scale_exponent).size + np.asarray(rhs.scale_exponent).size),
            "product_address_generation_proxy": int(np.asarray(descriptors.nonzero).sum()),
            "final_output_materializations_or_requantizations_proxy": trace.dot_count,
            "requantization_note": "The reference leaves final output-format choice open; this counts it as one non-free boundary action per dot output.",
            "decoded_product_vs_descriptor_max_abs_gap": float(
                np.max(np.abs(decoded_dot - np.asarray([dot["immediate_value"] for dot in analysis.per_dot])))
            ),
            "descriptor_has_final_address": bool(hasattr(descriptors, "address")),
            "descriptor_fields": [
                "sign",
                "product_exponent",
                "phase",
                "address",
                "lhs_scale_exponent",
                "rhs_scale_exponent",
            ],
        }
    )
    return analysis, row, lhs, rhs


def evaluate_numeric_baselines(
    traces: Sequence[OperandTrace], *, group_size: int | None) -> list[dict[str, object]]:
    """FP16/BF16/block-INT8 baselines with no invented address semantics."""
    rows: list[dict[str, object]] = []
    for trace in traces:
        rows.append(
            _trace_baseline_row(
                trace,
                representation="FP16",
                left=np.asarray(trace.a, dtype=np.float16).astype(np.float32),
                right=np.asarray(trace.b, dtype=np.float16).astype(np.float32),
                group_size=None,
                element_bits=16,
                classification="numerical_baseline",
            )
        )
        rows.append(
            _trace_baseline_row(
                trace,
                representation="BF16",
                left=_bf16_round_trip(trace.a),
                right=_bf16_round_trip(trace.b),
                group_size=None,
                element_bits=16,
                classification="numerical_baseline",
            )
        )
        effective_group = _effective_group_size(trace, group_size)
        left, _ = block_int8_grouped(trace.a, effective_group)
        right, _ = block_int8_grouped(trace.b, effective_group)
        rows.append(
            _trace_baseline_row(
                trace,
                representation="block_power_of_two_INT8",
                left=left,
                right=right,
                group_size=effective_group,
                element_bits=8,
                classification="numerical_baseline",
            )
        )
    return rows


def baseline_specs() -> tuple[tuple[str, ClosedLogSpec, str], ...]:
    """Keep direct CurveFP and #148 endpoint-cover policies distinct.

    The closed product algebra, finite phases, signed counts, and Kulisch
    semantics are CurveFP reproductions.  These operand traces deliberately
    use one dynamic ``ceil(absmax)`` scale policy for *both* activations and
    weights; CurveFP describes a separate MSE scale search for static weights.
    Therefore the whole quantization policy is labelled an engineering
    variation even for published CurveFP E/C layouts.  The legacy #148
    endpoint-cover policy remains a separate engineering baseline.
    """
    return (
        (
            "CurveFP_7b_E3_C3_binary_ceil_absmax",
            ClosedLogSpec(E=3, C=3, p=1, q=1, scale_policy="curvefp_ceil_absmax"),
            "curvefp_algebra_reproduction__dynamic_scale_policy_engineering_variation",
        ),
        (
            "CurveFP_published_8b_E4_C3_forward_ceil_absmax",
            ClosedLogSpec(E=4, C=3, p=1, q=1, scale_policy="curvefp_ceil_absmax"),
            "curvefp_algebra_reproduction__dynamic_scale_policy_engineering_variation",
        ),
        (
            "CurveFP_published_8b_E5_C2_gradient_context_ceil_absmax",
            ClosedLogSpec(E=5, C=2, p=1, q=1, scale_policy="curvefp_ceil_absmax"),
            "curvefp_algebra_reproduction__dynamic_scale_policy_engineering_variation",
        ),
        (
            "ClosedLog_148layout_8b_E3_C4_binary_ceil_absmax",
            ClosedLogSpec(E=3, C=4, p=1, q=1, scale_policy="curvefp_ceil_absmax"),
            "engineering_variation_148_E3_C4_layout_with_curvefp_scale",
        ),
        (
            "ClosedLog_148carry_7b_E3_C3_binary_endpoint_cover",
            ClosedLogSpec(E=3, C=3, p=1, q=1, scale_policy="endpoint_cover"),
            "engineering_variation_endpoint_cover_148_carry_forward",
        ),
        (
            "ClosedLog_148carry_8b_E3_C4_binary_endpoint_cover",
            ClosedLogSpec(E=3, C=4, p=1, q=1, scale_policy="endpoint_cover"),
            "engineering_variation_endpoint_cover_148_carry_forward",
        ),
    )


def emit_strongest_baseline_product_address_dataset(
    traces: Sequence[OperandTrace], *, group_size: int | None, directory: Path
) -> Path:
    """Emit #146's directly reusable product-address dataset once per trace.

    This deliberately uses the strongest completed-#148 continuation baseline
    (8-bit E3C4 binary with its endpoint-cover scale policy).  It is not a
    claim that the policy is CurveFP's static-weight calibration; the manifest
    records that distinction.  All later analyses can load the descriptor
    shards without invoking Torch/Transformers or recapturing model operands.
    """
    label, spec, classification = baseline_specs()[-1]
    entries: list[dict[str, object]] = []
    for trace in traces:
        effective_group = _effective_group_size(trace, group_size) if group_size is not None else None
        lhs = quantize_grouped(trace.a, spec, group_size=effective_group)
        rhs = quantize_grouped(trace.b, spec, group_size=effective_group)
        descriptors = product_descriptors(lhs, rhs)
        entries.append(
            save_product_address_trace(
                directory,
                trace=trace,
                representation=label,
                descriptors=descriptors,
                metadata={
                    "issue": "#146",
                    "classification": classification,
                    "group_size": effective_group if effective_group is not None else "tensorwise",
                    "scale_policy_note": "Dynamic ceil/endpoint block scales applied to both operands; static-weight MSE calibration is not reproduced here.",
                    "curvefp_reference": CURVEFP_REFERENCE,
                },
            )
        )
    return write_product_address_manifest(
        directory,
        entries,
        generation={
            "command": "python -m domain_scaling_lab.closed_log_experiments --sections baseline --baseline-group-size 64",
            "baseline_group_size": group_size if group_size is not None else "tensorwise",
            "purpose": "#146 direct product descriptor substrate; phase plus product_exponent is the accumulator address.",
            "curvefp_reference": CURVEFP_REFERENCE,
        },
    )


def run_curvefp_baselines(
    traces: Sequence[OperandTrace], *, group_size: int | None
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, str], TraceAnalysis],
]:
    """#146's common product-address substrate and CurveFP reproduction rows."""
    trace_rows: list[dict[str, object]] = []
    per_dot_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    exponent_rows: list[dict[str, object]] = []
    analyses: dict[tuple[str, str], TraceAnalysis] = {}
    for label, spec, classification in baseline_specs():
        for trace in traces:
            analysis, row, _, _ = evaluate_closed_log_trace(
                trace,
                spec=spec,
                group_size=group_size,
                label=label,
                classification=classification,
                issue="#146",
            )
            analyses[(label, trace.trace_id)] = analysis
            trace_rows.append(row)
            context = {
                "trace_id": trace.trace_id,
                "representation": label,
                "group_size": _effective_group_size(trace, group_size) if group_size is not None else "tensorwise",
                "E": spec.E,
                "C": spec.C,
                "p": spec.p,
                "q": spec.q,
                "H": spec.H,
                "issue": "#146",
                "classification": classification,
                "scale_policy": spec.scale_policy,
            }
            per_dot_rows.extend({**context, **item} for item in analysis.per_dot)
            phase_rows.extend({**context, **item} for item in analysis.phase_rows)
            state_rows.extend({**context, **item} for item in analysis.state_rows)
            exponent_rows.extend({**context, **item} for item in analysis.exponent_rows)
    return trace_rows, per_dot_rows, phase_rows, state_rows, exponent_rows, analyses


def _even_sample(values: np.ndarray, limit: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    if limit < 1 or len(flat) <= limit:
        return flat.copy()
    positions = np.linspace(0, len(flat) - 1, limit, dtype=np.int64)
    return flat[positions]


def scalar_sources(
    traces: Sequence[OperandTrace], *, synthetic_count: int, sample_values: int, seed: int
) -> list[tuple[str, str, np.ndarray, dict[str, object]]]:
    """Retained synthetic distributions plus role-labelled transformer operands."""
    sources: list[tuple[str, str, np.ndarray, dict[str, object]]] = []
    for name, values in synthetic_distributions(synthetic_count, 145).items():
        sources.append((f"synthetic:{name}", "synthetic", _even_sample(values, sample_values), {"distribution": name, "seed": 145}))
    for trace in traces:
        for role, values in (("activation", trace.a), ("weight", trace.b)):
            source_name = f"{trace.trace_id}:{role}"
            sources.append(
                (
                    source_name,
                    trace.source,
                    _even_sample(values, sample_values),
                    {**_trace_context(trace), "operand_role": role, "sampling": "evenly spaced flattening", "sample_limit": sample_values, "seed": seed},
                )
            )
    return sources


def continuous_radices(points: int) -> list[float]:
    if points < 2:
        raise ValueError("continuous radix sweep needs at least two grid points")
    values = list(np.linspace(1.1, 4.0, points))
    values.extend((math.sqrt(2.0), 2.0, math.e, 4.0))
    return sorted({round(float(value), 12) for value in values})


def _layout_for_total_bits(total_bits: int) -> tuple[int, int]:
    """Match the surviving #148-style balanced E/C starting layouts."""
    if total_bits < 4:
        raise ValueError("at least sign + one E + one C bit is required")
    E = (total_bits - 1) // 2
    C = total_bits - 1 - E
    return E, C


def _continuous_radix_coordinate_control(
    source: tuple[str, str, np.ndarray, dict[str, object]],
    *,
    E: int,
    C: int,
    element_bits: int,
    group_size: int,
    scale_policy: str,
) -> dict[str, object]:
    """Evaluate the fixed log2-vs-ln coordinate identity control at radix e."""
    source_id, source_kind, values, metadata = source
    effective_group = min(group_size, len(values))
    log2_encoding = quantize_generic_log_grouped(
        values,
        E=E,
        C=C,
        radix=math.e,
        group_size=effective_group,
        log_kind="log2",
        scale_policy=scale_policy,
    )
    ln_encoding = quantize_generic_log_grouped(
        values,
        E=E,
        C=C,
        radix=math.e,
        group_size=effective_group,
        log_kind="ln",
        scale_policy=scale_policy,
    )
    return {
        "issue": "#155",
        "classification": "mathematical_equivalence_control_not_a_mechanism",
        "source_id": source_id,
        "source": source_kind,
        "E": E,
        "C": C,
        "element_bits": element_bits,
        "scale_policy": scale_policy,
        "radix": math.e,
        "grid": "identical r^(n/K) grid; nearest absolute magnitude selection",
        "log2_vs_ln_code_disagreement_rate": float(np.mean(log2_encoding.codes != ln_encoding.codes)),
        "log2_vs_ln_reconstruction_max_abs_gap": float(
            np.max(np.abs(log2_encoding.decoded.astype(np.float64) - ln_encoding.decoded.astype(np.float64)))
        ),
        **metadata,
    }


def run_continuous_radix_sweep(
    sources: Sequence[tuple[str, str, np.ndarray, dict[str, object]]],
    *,
    total_bits: Sequence[int],
    group_size: int,
    points: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """#155 scalar-only arbitrary-radix sweep, including the ln/log2 control."""
    rows: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for bits in total_bits:
        E, C = _layout_for_total_bits(int(bits))
        for scale_policy in ("curvefp_ceil_absmax", "endpoint_cover"):
            policy_classification = (
                "curvefp_grid_algebra_reproduction__dynamic_scale_policy_engineering_variation"
                if scale_policy == "curvefp_ceil_absmax"
                else "engineering_variation_endpoint_cover_148_carry_forward"
            )
            for source_id, source_kind, values, metadata in sources:
                for radix in continuous_radices(points):
                    encoding = quantize_generic_log_grouped(
                        values,
                        E=E,
                        C=C,
                        radix=radix,
                        group_size=min(group_size, len(values)),
                        log_kind="log2",
                        scale_policy=scale_policy,
                    )
                    rows.append(
                        {
                            "issue": "#155",
                            "classification": policy_classification,
                            "source_id": source_id,
                            "source": source_kind,
                            "radix_label": _radix_label(radix),
                            "grid_coordinate": "r^(n/K), nearest absolute reconstructed magnitude",
                            **metadata,
                            **_generic_scalar_metrics(
                                values,
                                encoding,
                                E=E,
                                C=C,
                                radix=radix,
                                group_size=min(group_size, len(values)),
                                scale_policy=scale_policy,
                            ),
                        }
                    )
                # Coordinate identities are tested on the same real grid.
                # Exact e is intentionally chosen because it makes the
                # distinction from a rational closed layout especially visible.
                controls.append(
                    _continuous_radix_coordinate_control(
                        (source_id, source_kind, values, metadata),
                        E=E,
                        C=C,
                        element_bits=int(bits),
                        group_size=group_size,
                        scale_policy=scale_policy,
                    )
                )
    return rows, controls


def _radix_label(radix: float) -> str:
    special = {
        round(math.sqrt(2.0), 12): "sqrt2",
        round(2.0, 12): "2",
        round(math.e, 12): "e",
        round(4.0, 12): "4",
    }
    return special.get(round(float(radix), 12), "continuous")


def rational_radix_candidates(
    *,
    max_q: int,
    minimum_radix: float = 1.1,
    maximum_radix: float = 4.0,
) -> list[Fraction]:
    """All reduced p/q candidates in the requested radix interval, q <= max_q."""
    if max_q < 1:
        raise ValueError("max_q must be positive")
    minimum_alpha = math.log2(minimum_radix)
    maximum_alpha = math.log2(maximum_radix)
    candidates: set[Fraction] = set()
    for q in range(1, max_q + 1):
        for p in range(max(1, math.ceil(minimum_alpha * q)), math.floor(maximum_alpha * q) + 1):
            fraction = Fraction(p, q)
            if minimum_alpha <= float(fraction) <= maximum_alpha:
                candidates.add(fraction)
    return sorted(candidates, key=lambda value: (float(value), value.denominator, value.numerator))


def _select_rational_candidates(candidates: Sequence[Fraction], limit: int) -> list[Fraction]:
    """Optionally shrink a runnable view while retaining specified landmark grids.

    A nonpositive ``limit`` means the full q-bounded search.  The output notes
    selection explicitly, so an abbreviated run cannot masquerade as an
    exhaustive p/q result.
    """
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    required = {Fraction(1, 2), Fraction(1, 1), Fraction(2, 1)}
    target_e = math.log2(math.e)
    required.add(min(candidates, key=lambda item: abs(float(item) - target_e)))
    selected = {value for value in required if value in candidates}
    available = [value for value in candidates if value not in selected]
    remaining = max(0, limit - len(selected))
    if remaining:
        positions = np.linspace(0, len(available) - 1, remaining, dtype=np.int64)
        selected.update(available[index] for index in positions)
    return sorted(selected, key=lambda value: (float(value), value.denominator, value.numerator))


def _select_traces(traces: Sequence[OperandTrace], limit: int) -> list[OperandTrace]:
    if limit <= 0 or len(traces) <= limit:
        return list(traces)
    # Preserve at least one genuine transformer trace where it is available;
    # then fill reproducibly across all widths/sources rather than picking one
    # convenient BERT tensor.
    ordered = sorted(traces, key=lambda trace: (trace.source, trace.width, trace.trace_id))
    real = [trace for trace in ordered if trace.source == "pretrained_transformer"]
    synthetic = [trace for trace in ordered if trace.source != "pretrained_transformer"]
    chosen: list[OperandTrace] = []
    if real:
        chosen.append(real[len(real) // 2])
    if synthetic and len(chosen) < limit:
        chosen.append(synthetic[len(synthetic) // 2])
    pool = [trace for trace in ordered if trace not in chosen]
    remaining = limit - len(chosen)
    if remaining > 0:
        positions = np.linspace(0, len(pool) - 1, remaining, dtype=np.int64)
        chosen.extend(pool[index] for index in positions)
    return sorted({trace.trace_id: trace for trace in chosen}.values(), key=lambda trace: trace.trace_id)


def _near_e_rank(candidates: Sequence[Fraction]) -> dict[Fraction, int]:
    target = math.log2(math.e)
    return {candidate: index + 1 for index, candidate in enumerate(sorted(candidates, key=lambda item: (abs(float(item) - target), item.denominator, item.numerator)))}


def _rational_scalar_rows(
    sources: Sequence[tuple[str, str, np.ndarray, dict[str, object]]],
    spec: ClosedLogSpec,
    *,
    group_size: int,
    classification: str,
    near_e_rank: int,
    log2_radix: float,
    scale_policy: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_id, source_kind, values, metadata in sources:
        effective_group = min(group_size, len(values))
        metrics, _, _ = _core_scalar_metrics(values, spec=spec, group_size=effective_group)
        rows.append(
            {
                "issue": "#155/#152",
                "classification": classification,
                "source_id": source_id,
                "source": source_kind,
                "candidate_scope": "full_scalar_q_bounded_search",
                "near_e_rank": near_e_rank,
                "log2_radix": log2_radix,
                "scale_policy": scale_policy,
                **metadata,
                **metrics,
            }
        )
    return rows


def _rational_product_trace_rows(
    trace: OperandTrace,
    spec: ClosedLogSpec,
    *,
    group_size: int,
    label: str,
    classification: str,
    candidate_scope: str,
    near_e_rank: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    analysis, row, _, _ = evaluate_closed_log_trace(
        trace,
        spec=spec,
        group_size=group_size,
        label=label,
        classification=classification,
        issue="#152/#155",
        validate_kulisch=False,
        compute_exact_reachability=False,
        compute_temporal_locality=False,
    )
    log2_radix = float(Fraction(spec.p, spec.q))
    row.update(
        {
            "candidate_scope": candidate_scope,
            "near_e_rank": near_e_rank,
            "log2_radix": log2_radix,
            "scale_policy": spec.scale_policy,
            "occupied_product_bins": row["mean_unique_addresses_per_dot"],
            "product_address_entropy_bits": row["address_entropy_bits"],
            "kulisch_validation_executed": False,
            "exact_reachability_enumerated": False,
            "temporal_locality_enumerated": False,
        }
    )
    phase_rows = [
        {
            "issue": "#152/#155",
            "classification": classification,
            "trace_id": trace.trace_id,
            "E": spec.E,
            "C": spec.C,
            "p": spec.p,
            "q": spec.q,
            "H": spec.H,
            "radix": spec.radix,
            "scale_policy": spec.scale_policy,
            "near_e_rank": near_e_rank,
            **phase_row,
        }
        for phase_row in analysis.phase_rows
    ]
    return row, phase_rows


def run_rational_radix_sweep(
    sources: Sequence[tuple[str, str, np.ndarray, dict[str, object]]],
    traces: Sequence[OperandTrace],
    *,
    total_bits: Sequence[int],
    group_size: int,
    max_q: int,
    product_candidate_limit: int,
    product_trace_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Shared #152/#155 rational-radix data path.

    The scalar rows cover every q-bounded candidate.  Product rows can be
    limited only through an explicit CLI knob.  The #146 baseline performs the
    exact O(H) Kulisch validation; this sweep deliberately uses the address-only
    path so every selected radix is screened on the same occupancy statistics.
    The summary records both the full candidate count and product subset count.
    """
    all_candidates = rational_radix_candidates(max_q=max_q)
    product_candidates = _select_rational_candidates(all_candidates, product_candidate_limit)
    product_traces = _select_traces(traces, product_trace_limit)
    ranks = _near_e_rank(all_candidates)
    scalar_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    product_phase_rows: list[dict[str, object]] = []

    for bits in total_bits:
        E, C = _layout_for_total_bits(int(bits))
        for scale_policy in ("curvefp_ceil_absmax", "endpoint_cover"):
            classification = (
                "curvefp_rational_phase_algebra_reproduction__dynamic_scale_policy_engineering_variation"
                if scale_policy == "curvefp_ceil_absmax"
                else "engineering_variation_endpoint_cover_rational_grid"
            )
            for candidate in all_candidates:
                spec = ClosedLogSpec(
                    E=E, C=C, p=candidate.numerator, q=candidate.denominator, scale_policy=scale_policy
                )
                scalar_rows.extend(
                    _rational_scalar_rows(
                        sources,
                        spec,
                        group_size=group_size,
                        classification=classification,
                        near_e_rank=ranks[candidate],
                        log2_radix=float(candidate),
                        scale_policy=scale_policy,
                    )
                )
            for candidate in product_candidates:
                spec = ClosedLogSpec(
                    E=E, C=C, p=candidate.numerator, q=candidate.denominator, scale_policy=scale_policy
                )
                for trace in product_traces:
                    row, phase_rows = _rational_product_trace_rows(
                        trace,
                        spec,
                        group_size=group_size,
                        label=f"rational_2^({candidate.numerator}/{candidate.denominator})_{bits}b_{scale_policy}",
                        classification=classification,
                        candidate_scope=(
                            "full_product_q_bounded_search"
                            if product_candidate_limit <= 0
                            else "explicit_product_subset"
                        ),
                        near_e_rank=ranks[candidate],
                    )
                    product_rows.append(row)
                    product_phase_rows.extend(phase_rows)
    scope = {
        "max_q": max_q,
        "full_rational_candidate_count": len(all_candidates),
        "product_candidate_count": len(product_candidates),
        "product_trace_count": len(product_traces),
        "product_candidate_scope": "full" if product_candidate_limit <= 0 else "explicit_subset",
        "exact_e_note": "e is scalar-only: log2(e) is irrational, so exact e has no finite exact binary rational phase decomposition.",
        "nearest_q_bounded_e_candidate": {
            "p": min(all_candidates, key=lambda item: abs(float(item) - math.log2(math.e))).numerator,
            "q": min(all_candidates, key=lambda item: abs(float(item) - math.log2(math.e))).denominator,
        },
    }
    return scalar_rows, product_rows, product_phase_rows, scope


def _rational_aggregate_row(
    key: tuple[int, int, int, int, str],
    scalar_group: Sequence[Mapping[str, object]],
    products: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    E, C, p, q, scale_policy = key
    candidate = Fraction(p, q)
    return {
        "E": E,
        "C": C,
        "p": p,
        "q": q,
        "scale_policy": scale_policy,
        "radix": float(math.exp2(float(candidate))),
        "log2_radix": float(candidate),
        "H": q * (1 << C) // math.gcd(p, q * (1 << C)),
        "mean_scalar_nmse": float(np.mean([float(item["scalar_nmse"]) for item in scalar_group])),
        "max_scalar_nmse": float(np.max([float(item["scalar_nmse"]) for item in scalar_group])),
        "mean_scalar_clipping_rate": float(np.mean([float(item["clipping_rate"]) for item in scalar_group])),
        "scalar_source_count": len(scalar_group),
        "product_trace_count": len(products),
        "mean_occupied_product_bins": (
            float(np.mean([float(item["occupied_product_bins"]) for item in products]))
            if products
            else float("nan")
        ),
        "mean_duplicate_address_rate": (
            float(np.mean([float(item["aggregate_duplicate_address_rate"]) for item in products]))
            if products
            else float("nan")
        ),
        "mean_product_address_entropy_bits": (
            float(np.mean([float(item["product_address_entropy_bits"]) for item in products]))
            if products
            else float("nan")
        ),
        "mean_product_exponent_span": (
            float(np.mean([float(item["product_exponent_span"]) for item in products]))
            if products
            else float("nan")
        ),
        "mean_phase_event_entropy_bits": (
            float(np.mean([float(item["phase_event_entropy_bits"]) for item in products]))
            if products
            else float("nan")
        ),
        "classification": "engineering_variation_radix_pareto",
    }

def aggregate_rational_rows(
    scalar_rows: Sequence[Mapping[str, object]], product_rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Join scalar quality and address cost for #152 Pareto/conditional-H tests."""
    scalar_by_key: dict[tuple[int, int, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    product_by_key: dict[tuple[int, int, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in scalar_rows:
        scalar_by_key[(int(row["E"]), int(row["C"]), int(row["p"]), int(row["q"]), str(row["scale_policy"]))].append(row)
    for row in product_rows:
        product_by_key[(int(row["E"]), int(row["C"]), int(row["p"]), int(row["q"]), str(row["scale_policy"]))].append(row)
    aggregate = [
        _rational_aggregate_row(key, scalar_group, product_by_key.get(key, []))
        for key, scalar_group in scalar_by_key.items()
    ]

    # Error/H and error/address fronts only make sense at fixed element layout
    # and scale policy.  A global front would let an unrelated E/C or scaling
    # policy erase a valid design trade-off in the format being examined.
    pareto_groups: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in aggregate:
        pareto_groups[(int(row["E"]), int(row["C"]), str(row["scale_policy"]))].append(row)
    scalar_h_pareto: list[dict[str, object]] = []
    address_pareto: list[dict[str, object]] = []
    for (E, C, scale_policy), group in sorted(pareto_groups.items()):
        for row in pareto_front(group, minimize=("mean_scalar_nmse", "H")):
            row["pareto_objective"] = "scalar_error_vs_H"
            row["pareto_scope"] = f"E{E}_C{C}_{scale_policy}"
            scalar_h_pareto.append(row)
        for row in pareto_front(group, minimize=("mean_scalar_nmse", "mean_occupied_product_bins")):
            row["pareto_objective"] = "scalar_error_vs_occupied_product_bins"
            row["pareto_scope"] = f"E{E}_C{C}_{scale_policy}"
            address_pareto.append(row)
    diagnostic = _address_cost_diagnostic(aggregate)
    return aggregate, scalar_h_pareto + address_pareto, diagnostic

def _address_cost_diagnostic(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Test whether observed addresses vary once theoretical H is held fixed."""
    usable = [row for row in rows if math.isfinite(float(row["mean_occupied_product_bins"]))]
    metrics = ("mean_occupied_product_bins", "mean_duplicate_address_rate", "mean_product_address_entropy_bits")
    correlations = {metric: pearson_r([float(row["H"]) for row in usable], [float(row[metric]) for row in usable]) for metric in metrics}
    by_layout_h: dict[tuple[int, int, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in usable:
        by_layout_h[(int(row["E"]), int(row["C"]), str(row["scale_policy"]), int(row["H"]))].append(row)
    conditional: dict[str, float] = {}
    groups = 0
    for metric in metrics:
        ranges = []
        for items in by_layout_h.values():
            if len(items) < 2:
                continue
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            ranges.append(float(values.max() - values.min()))
        conditional[metric] = float(np.mean(ranges)) if ranges else 0.0
        groups = max(groups, len(ranges))
    # This is a predeclared evidence flag, not a hardware conclusion.  It says
    # only whether p/q candidates sharing H visibly differ in measured address
    # structure at this trace sample.
    signal = conditional["mean_occupied_product_bins"] > 0.5 or conditional["mean_duplicate_address_rate"] > 0.01
    return {
        "issue": "#152",
        "classification": "empirical_address_cost_test",
        "usable_candidate_layouts": len(usable),
        "within_H_comparison_groups": groups,
        "H_correlations": correlations,
        "mean_within_H_metric_ranges": conditional,
        "predeclared_signal_rule": "mean occupied-bin range > 0.5 OR duplicate-rate range > 0.01 among candidates with the same E/C/H",
        "empirical_address_behavior_beyond_H_at_this_sample": signal,
        "interpretation": "candidate evidence" if signal else "no measured additional address axis at this sample",
    }


def run_cheap_trace_analyses(
    analyses: Mapping[tuple[str, str], TraceAnalysis],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """#150/#151/#149 analyses using exactly the #146 baseline descriptors."""
    coalescing_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    approximation_rows: list[dict[str, object]] = []
    associative_rows: list[dict[str, object]] = []
    for (representation, trace_id), analysis in analyses.items():
        context = {
            "trace_id": trace_id,
            "representation": representation,
            "E": analysis.summary.get("E"),
            "C": analysis.summary.get("C"),
            "p": analysis.summary.get("p"),
            "q": analysis.summary.get("q"),
            "H": analysis.summary.get("phase_count"),
            "scale_policy": analysis.summary.get("scale_policy"),
        }
        coalescing_rows.extend({"issue": "#150", **context, **row} for row in lane_coalescing_analysis(analysis, DEFAULT_LANE_SIZES))
        phase_rows.extend(
            {
                "issue": "#151",
                "classification": "curvefp_phase_usage_reproduction",
                **context,
                **row,
                "frequency_vs_absolute_contribution_gap": float(row["event_fraction"] - row["absolute_contribution_fraction"]),
            }
            for row in analysis.phase_rows
        )
        approximation_rows.extend({"issue": "#151", **context, **row} for row in phase_specialization_analysis(analysis))
        associative_rows.extend(
            {"issue": "#149", **context, **row}
            for row in associative_accumulator_analysis(analysis, DEFAULT_ASSOCIATIVE_CAPACITIES)
        )
    return coalescing_rows, phase_rows, approximation_rows, associative_rows


def run_group_size_sweep(
    traces: Sequence[OperandTrace], *, group_sizes: Sequence[int | None]
) -> list[dict[str, object]]:
    """#153: use the same datatype/descriptor path while only changing scale scope."""
    rows: list[dict[str, object]] = []
    for label, spec, baseline_classification in baseline_specs():
        for trace in traces:
            for requested_group in group_sizes:
                analysis, row, _, _ = evaluate_closed_log_trace(
                    trace,
                    spec=spec,
                    group_size=requested_group,
                    label=label,
                    classification="engineering_variation_scale_group_size",
                    issue="#153",
                )
                row.update(
                    {
                        "requested_group_size": requested_group if requested_group is not None else "tensorwise",
                        "effective_group_size": _effective_group_size(trace, requested_group)
                        if requested_group is not None
                        else "tensorwise",
                        "accumulator_live_state_footprint_proxy": row["mean_unique_addresses_per_dot"],
                        "phase_distribution_entropy_bits": row["phase_event_entropy_bits"],
                        "classification": "engineering_variation_scale_group_size",
                        "baseline_scale_policy_classification": baseline_classification,
                        "scale_policy": spec.scale_policy,
                        "issue": "#153",
                    }
                )
                rows.append(row)
    return rows


def practical_layouts(total_bits: int) -> list[tuple[int, int]]:
    """All valid E/C splits at fixed width, including the K=1 single-curve control."""
    if total_bits < 4:
        raise ValueError("at least 4 total bits are needed for a useful sign + E/C layout")
    return [(E, total_bits - 1 - E) for E in range(1, total_bits)]


def run_layout_sweep(
    traces: Sequence[OperandTrace], *, total_bits: Sequence[int], group_size: int
) -> list[dict[str, object]]:
    """#154: enumerate valid E/C layouts without changing any other mechanism."""
    rows: list[dict[str, object]] = []
    for bits in total_bits:
        for E, C in practical_layouts(int(bits)):
            for scale_policy in ("curvefp_ceil_absmax", "endpoint_cover"):
                spec = ClosedLogSpec(E=E, C=C, p=1, q=1, scale_policy=scale_policy)
                classification = (
                    "engineering_variation_E_C_layout_curvefp_scale"
                    if scale_policy == "curvefp_ceil_absmax"
                    else "engineering_variation_E_C_layout_endpoint_cover"
                )
                for trace in traces:
                    analysis, row, _, _ = evaluate_closed_log_trace(
                        trace,
                        spec=spec,
                        group_size=group_size,
                        label=f"ClosedLog_{bits}b_E{E}_C{C}_binary_{scale_policy}",
                        classification=classification,
                        issue="#154",
                    )
                    row.update(
                        {
                            "issue": "#154",
                            "classification": classification,
                            "total_element_bits": bits,
                            "layout": f"E{E}_C{C}",
                            "scale_policy": scale_policy,
                            "accumulator_state_cost_proxy": row["mean_unique_addresses_per_dot"],
                        }
                    )
                    rows.append(row)
    return rows


def _format_role(row: Mapping[str, object]) -> str:
    module = str(row.get("module") or "")
    if "attention.self" in module:
        return "attention_projection"
    if "attention.output" in module:
        return "attention_output"
    if "intermediate" in module:
        return "mlp_expand"
    if module.endswith("output.dense"):
        return "mlp_contract"
    return str(row.get("source") or "unknown")


def format_selection_analysis(layout_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compare global and controlled adaptive layout policies with charged state.

    This is an in-sample selection screen, not a claim that a new adaptive
    format generalizes.  The selection/control metadata proxy is explicit so a
    lower numerical error cannot hide a switching mechanism.
    """
    results: list[dict[str, object]] = []
    by_bits: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in layout_rows:
        by_bits[(int(row["total_element_bits"]), str(row["scale_policy"]))].append(row)
    policies = {
        "one_global_format": lambda row: "all",
        "role_specific_format": _format_role,
        "layer_specific_format": lambda row: f"layer:{row.get('layer')}",
        "tensor_class_specific_format": lambda row: str(row.get("trace_id")),
    }
    for (bits, scale_policy), rows in by_bits.items():
        layouts = sorted({str(row["layout"]) for row in rows})
        format_id_bits = max(1, math.ceil(math.log2(max(2, len(layouts)))))
        for policy_name, grouping in policies.items():
            groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in rows:
                groups[grouping(row)].append(row)
            selected_layouts: dict[str, str] = {}
            chosen_rows: list[Mapping[str, object]] = []
            for name, items in groups.items():
                candidates: dict[str, list[Mapping[str, object]]] = defaultdict(list)
                for item in items:
                    candidates[str(item["layout"])].append(item)
                winner = min(
                    candidates,
                    key=lambda layout: (
                        float(np.mean([float(item["quantized_dot_nmse"]) for item in candidates[layout]])),
                        layout,
                    ),
                )
                selected_layouts[name] = winner
                chosen_rows.extend(candidates[winner])
            unique_formats = sorted(set(selected_layouts.values()))
            # A compact 32-bit descriptor per distinct layout plus a format ID
            # per controlled tensor/layer/role is intentionally conservative.
            control_bits = len(unique_formats) * 32 + len(groups) * format_id_bits
            results.append(
                {
                    "issue": "#154",
                    "classification": "adaptive_layout_selection_screen",
                    "total_element_bits": bits,
                    "scale_policy": scale_policy,
                    "policy": policy_name,
                    "controlled_units": len(groups),
                    "available_layouts": layouts,
                    "selected_layouts": selected_layouts,
                    "unique_selected_layout_count": len(unique_formats),
                    "format_id_bits_per_controlled_unit": format_id_bits,
                    "format_selection_metadata_bits_proxy": control_bits,
                    "mean_quantized_dot_nmse": float(np.mean([float(row["quantized_dot_nmse"]) for row in chosen_rows])),
                    "mean_occupied_product_bins": float(
                        np.mean([float(row["mean_unique_addresses_per_dot"]) for row in chosen_rows])
                    ),
                    "mean_phase_count_H": float(np.mean([float(row["H"]) for row in chosen_rows])),
                    "selection_note": "In-sample lower bound only; validate selected policies on held-out models/tensors before treating as robust.",
                }
            )
    return results


def _coalescing_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["writes_avoided_fraction"]))
    return {
        "issue": "#150",
        "screen": "pre_routing_duplicate_coalescing",
        "best_writes_avoided_fraction": best["writes_avoided_fraction"],
        "best_address_coalescing_ratio": best["address_coalescing_ratio"],
        "best_effective_write_coalescing_ratio": best["effective_write_coalescing_ratio"],
        "required_accumulator_update_cost_for_break_even_simple_ops": best[
            "break_even_update_cost_in_simple_op_units"
        ],
        "conclusion_rule": (
            "deprioritize if collision savings cannot exceed explicitly reported "
            "compare/shuffle/popcount cost"
        ),
    }


def _phase_frequency_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    contribution_gap = [abs(float(row["frequency_vs_absolute_contribution_gap"])) for row in rows]
    return {
        "issue": "#151",
        "screen": "phase_frequency_vs_numerical_contribution",
        "max_absolute_frequency_contribution_gap": float(max(contribution_gap)),
        "conclusion_rule": (
            "frequency-only omission is invalid whenever rare phases retain material absolute "
            "contribution; use approximation NMSE rows."
        ),
    }


def _phase_approximation_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    # A budget smaller than every phase correctly selects no phase.  Those
    # no-op controls must not be reported as safe approximations.
    selected = [row for row in rows if bool(row.get("selected_any_phase"))]
    safe_omission = [row for row in selected if float(row["omission_nmse_vs_quantized_dot"]) <= 1e-4]
    safe_merge = [row for row in selected if float(row["merge_nmse_vs_quantized_dot"]) <= 1e-4]
    return {
        "issue": "#151",
        "screen": "controlled_rare_phase_omission",
        "tested_variants": len(rows),
        "no_op_frequency_budget_controls": len(rows) - len(selected),
        "actual_phase_omission_or_merge_variants": len(selected),
        "actual_omissions_at_or_below_1e-4_nmse": len(safe_omission),
        "actual_merges_at_or_below_1e-4_nmse": len(safe_merge),
        "conclusion_rule": (
            "do not recommend omission/merge on frequency alone; no-op budgets are excluded "
            "and all selected approximations retain reported NMSE."
        ),
    }


def _associative_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    survivors = [
        row
        for row in rows
        if int(row["traffic_delta_vs_dense"]) <= 0
        and int(row["associative_storage_bits_proxy"]) < int(row["dense_histogram_counter_storage_bits_proxy"])
    ]
    return {
        "issue": "#149",
        "screen": "associative_accumulator_with_spills",
        "tested_capacity_points": len(rows),
        "capacity_points_beating_dense_on_storage_and_traffic_proxy": len(survivors),
        "conclusion_rule": (
            "falsify/deprioritize if no capacity beats dense proxy after tags, comparisons, "
            "spills, and final merges."
        ),
    }


def _group_size_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    by_key: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["representation"]), str(row["trace_id"]))].append(row)
    divergent = 0
    for grouped_rows in by_key.values():
        numerically_best = min(grouped_rows, key=lambda row: float(row["quantized_dot_nmse"]))
        systems_best = min(
            grouped_rows,
            key=lambda row: float(row["accumulator_live_state_footprint_proxy"]),
        )
        divergent += numerically_best["requested_group_size"] != systems_best["requested_group_size"]
    return {
        "issue": "#153",
        "screen": "numerical_vs_live_state_group_size_optimum",
        "trace_format_pairs": len(by_key),
        "pairs_with_different_numerical_and_live_state_optima": divergent,
        "conclusion_rule": "seek a joint Pareto improvement, not metadata savings alone.",
    }


def _layout_screening_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    global_rows = [row for row in rows if row["policy"] == "one_global_format"]
    adaptive_rows = [row for row in rows if row["policy"] != "one_global_format"]
    return {
        "issue": "#154",
        "screen": "adaptive_E_C_layouts",
        "global_policies": len(global_rows),
        "adaptive_policies": len(adaptive_rows),
        "conclusion_rule": (
            "only carry forward adaptive layouts if error/live-state benefit survives charged "
            "selection metadata and held-out validation."
        ),
    }


def issue_screening_summary(
    *,
    coalescing_rows: Sequence[Mapping[str, object]],
    phase_rows: Sequence[Mapping[str, object]],
    approximation_rows: Sequence[Mapping[str, object]],
    associative_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    layout_selection_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Predeclared cheap screens; they report evidence without redesigning criteria."""
    rows = (
        _coalescing_screening_row(coalescing_rows),
        _phase_frequency_screening_row(phase_rows),
        _phase_approximation_screening_row(approximation_rows),
        _associative_screening_row(associative_rows),
        _group_size_screening_row(group_rows),
        _layout_screening_row(layout_selection_rows),
    )
    return [row for row in rows if row is not None]

def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parsed


def _parse_sections(value: str) -> set[str]:
    valid = {"baseline", "radix", "cheap", "group", "layout"}
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = requested - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown sections: {', '.join(sorted(unknown))}")
    return requested or valid


def _trace_provenance(manifest: Mapping[str, object], traces: Sequence[OperandTrace]) -> dict[str, object]:
    entries = []
    for trace in traces:
        entries.append(
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "shape": [trace.dot_count, trace.width],
                "model": trace.metadata.get("model"),
                "revision": trace.metadata.get("revision"),
                "module": trace.metadata.get("module"),
                "layer": trace.metadata.get("layer"),
            }
        )
    return {
        "trace_format_version": manifest.get("format_version"),
        "capture_status": manifest.get("capture_status"),
        "trace_entries": entries,
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable closed-log experiment-run configuration."""

    trace_dir: Path
    output: Path
    sections: frozenset[str]
    capture: bool
    include_real: bool
    local_files_only: bool
    seed: int
    baseline_group_size: int
    group_sizes: tuple[int, ...]
    total_bits: tuple[int, ...]
    continuous_points: int
    rational_max_q: int
    product_candidate_limit: int
    product_trace_limit: int
    trace_limit: int
    scalar_sample_values: int
    synthetic_scalar_count: int


def _prepare_experiment_traces(
    config: ExperimentConfig,
) -> tuple[list[OperandTrace], Mapping[str, object], str]:
    config.output.mkdir(parents=True, exist_ok=True)
    all_traces, manifest, trace_action = ensure_operand_traces(
        config.trace_dir,
        capture=config.capture,
        include_real=config.include_real,
        local_files_only=config.local_files_only,
        seed=config.seed,
    )
    traces = _select_traces(all_traces, config.trace_limit)
    if not traces:
        raise ValueError("trace dataset is empty")
    return traces, manifest, trace_action


def _run_baseline_section(
    traces: Sequence[OperandTrace],
    config: ExperimentConfig,
    rows: dict[str, list[dict[str, object]]],
) -> tuple[dict[tuple[str, str], TraceAnalysis], Path | None]:
    if not {"baseline", "cheap"} & config.sections:
        return {}, None
    (
        trace_rows,
        per_dot_rows,
        phase_rows,
        state_rows,
        exponent_rows,
        analyses,
    ) = run_curvefp_baselines(traces, group_size=config.baseline_group_size)
    rows["146_curvefp_trace_summary"] = trace_rows
    rows["146_curvefp_per_dot"] = per_dot_rows
    rows["146_curvefp_phase_summary"] = phase_rows
    rows["146_curvefp_product_state_summary"] = state_rows
    rows["146_curvefp_exponent_summary"] = exponent_rows
    rows["146_numeric_baselines"] = evaluate_numeric_baselines(
        traces, group_size=config.baseline_group_size,
    )
    manifest = emit_strongest_baseline_product_address_dataset(
        traces,
        group_size=config.baseline_group_size,
        directory=config.trace_dir / "product_addresses",
    )
    return analyses, manifest


def _run_radix_section(
    traces: Sequence[OperandTrace],
    config: ExperimentConfig,
    rows: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, object], dict[str, object]]:
    if "radix" not in config.sections:
        return {}, {}
    sources = scalar_sources(
        traces,
        synthetic_count=config.synthetic_scalar_count,
        sample_values=config.scalar_sample_values,
        seed=config.seed,
    )
    continuous_rows, log_equivalence_rows = run_continuous_radix_sweep(
        sources,
        total_bits=config.total_bits,
        group_size=config.baseline_group_size,
        points=config.continuous_points,
    )
    rational_scalar_rows, rational_product_rows, rational_phase_rows, scope = run_rational_radix_sweep(
        sources,
        traces,
        total_bits=config.total_bits,
        group_size=config.baseline_group_size,
        max_q=config.rational_max_q,
        product_candidate_limit=config.product_candidate_limit,
        product_trace_limit=config.product_trace_limit,
    )
    aggregate_rows, pareto_rows, diagnostic = aggregate_rational_rows(
        rational_scalar_rows, rational_product_rows,
    )
    rows["155_continuous_scalar_radix"] = continuous_rows
    rows["155_ln_log2_control"] = log_equivalence_rows
    rows["152_155_rational_scalar_radix"] = rational_scalar_rows
    rows["152_155_rational_product_addresses"] = rational_product_rows
    rows["152_155_rational_phase_summary"] = rational_phase_rows
    rows["152_155_rational_aggregate"] = aggregate_rows
    rows["152_155_radix_pareto"] = pareto_rows
    return scope, diagnostic


@dataclass(frozen=True)
class ExperimentReport:
    config: ExperimentConfig
    rows: Mapping[str, list[dict[str, object]]]
    manifest: Mapping[str, object]
    traces: Sequence[OperandTrace]
    trace_action: str
    product_address_manifest: Path | None
    radix_scope: Mapping[str, object]
    address_diagnostic: Mapping[str, object]
    screening: list[dict[str, object]]


def _write_experiment_report(report: ExperimentReport) -> dict[str, object]:
    config = report.config
    for stem, table in report.rows.items():
        write_rows_csv(config.output / f"{stem}.csv", table)
    summary: dict[str, object] = {
        "runner": "domain_scaling_lab.closed_log_experiments",
        "curvefp_reference": CURVEFP_REFERENCE,
        "classification_legend": {
            "reproduction_validation": "CurveFP datatype, finite phase decomposition, signed histogram, and per-phase Kulisch behavior.",
            "engineering_variation": "A measured implementation/layout/radix choice, not a claim of a new CurveFP mechanism.",
            "new_mechanism": "Reserved for an effect that survives the predeclared screening criteria; this runner does not assert one automatically.",
        },
        "configuration": {
            "sections": sorted(config.sections),
            "seed": config.seed,
            "baseline_group_size": config.baseline_group_size,
            "group_sizes": list(config.group_sizes),
            "total_bits": list(config.total_bits),
            "continuous_radix_points": config.continuous_points,
            "rational_max_q": config.rational_max_q,
            "product_candidate_limit": config.product_candidate_limit,
            "product_trace_limit": config.product_trace_limit,
            "trace_limit": config.trace_limit,
            "scalar_sample_values": config.scalar_sample_values,
            "synthetic_scalar_count": config.synthetic_scalar_count,
            "trace_action": report.trace_action,
            "trace_dir": config.trace_dir,
            "product_address_manifest": report.product_address_manifest,
            "output": config.output,
        },
        "provenance": {
            **_trace_provenance(report.manifest, report.traces),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "command": " ".join(sys.argv),
            "external_dependencies": {
                "trace_capture": "torch + transformers are only required when --capture --include-real is requested",
                "pretrained_models": "not embedded in output; trace manifest records exact model revision where available",
            },
        },
        "row_counts": {stem: len(table) for stem, table in report.rows.items()},
        "radix_scope": dict(report.radix_scope),
        "empirical_address_cost_diagnostic": dict(report.address_diagnostic),
        "issue_screening": report.screening,
        "artifact_files": [f"{stem}.csv" for stem in sorted(report.rows)],
    }
    (config.output / "summary.json").write_text(
        json.dumps(_as_json(summary), indent=2, sort_keys=True) + "\n",
    )
    return summary


def run_experiments(
    *,
    trace_dir: Path,
    output: Path,
    sections: set[str],
    capture: bool = False,
    include_real: bool = True,
    local_files_only: bool = False,
    seed: int = DEFAULT_SEED,
    baseline_group_size: int = 64,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    total_bits: Sequence[int] = (7, 8),
    continuous_points: int = 31,
    rational_max_q: int = 16,
    product_candidate_limit: int = 0,
    product_trace_limit: int = 0,
    trace_limit: int = 0,
    scalar_sample_values: int = 8192,
    synthetic_scalar_count: int = 20_000,
) -> dict[str, object]:
    """Run requested phases and write compact, auditable artifacts.

    This is intentionally callable from a notebook/test without argparse.  A
    zero candidate/trace limit means exhaustive within the locally stored trace
    set.  The CLI's ``--quick`` mode changes these to explicit small limits.
    """
    config = ExperimentConfig(
        trace_dir=trace_dir,
        output=output,
        sections=frozenset(sections),
        capture=capture,
        include_real=include_real,
        local_files_only=local_files_only,
        seed=seed,
        baseline_group_size=baseline_group_size,
        group_sizes=tuple(group_sizes),
        total_bits=tuple(total_bits),
        continuous_points=continuous_points,
        rational_max_q=rational_max_q,
        product_candidate_limit=product_candidate_limit,
        product_trace_limit=product_trace_limit,
        trace_limit=trace_limit,
        scalar_sample_values=scalar_sample_values,
        synthetic_scalar_count=synthetic_scalar_count,
    )
    traces, manifest, trace_action = _prepare_experiment_traces(config)

    rows: dict[str, list[dict[str, object]]] = {}
    baseline_analyses, product_address_manifest = _run_baseline_section(
        traces, config, rows,
    )

    coalescing_rows: list[dict[str, object]] = []
    phase_usage_rows: list[dict[str, object]] = []
    phase_approximation_rows: list[dict[str, object]] = []
    associative_rows: list[dict[str, object]] = []
    if "cheap" in sections:
        coalescing_rows, phase_usage_rows, phase_approximation_rows, associative_rows = run_cheap_trace_analyses(
            baseline_analyses
        )
        rows["150_lane_coalescing"] = coalescing_rows
        rows["151_phase_usage"] = phase_usage_rows
        rows["151_phase_approximations"] = phase_approximation_rows
        rows["149_associative_screen"] = associative_rows

    radix_scope, address_diagnostic = _run_radix_section(
        traces, config, rows,
    )

    group_rows: list[dict[str, object]] = []
    if "group" in sections:
        group_rows = run_group_size_sweep(traces, group_sizes=tuple(group_sizes) + (None,))
        rows["153_group_size_sweep"] = group_rows

    layout_rows: list[dict[str, object]] = []
    layout_selection_rows: list[dict[str, object]] = []
    if "layout" in sections:
        layout_rows = run_layout_sweep(traces, total_bits=total_bits, group_size=baseline_group_size)
        layout_selection_rows = format_selection_analysis(layout_rows)
        rows["154_E_C_layout_sweep"] = layout_rows
        rows["154_layout_selection"] = layout_selection_rows

    screening = issue_screening_summary(
        coalescing_rows=coalescing_rows,
        phase_rows=phase_usage_rows,
        approximation_rows=phase_approximation_rows,
        associative_rows=associative_rows,
        group_rows=group_rows,
        layout_selection_rows=layout_selection_rows,
    )
    rows["issue_screening"] = screening

    return _write_experiment_report(
        ExperimentReport(
            config=config,
            rows=rows,
            manifest=manifest,
            traces=traces,
            trace_action=trace_action,
            product_address_manifest=product_address_manifest,
            radix_scope=radix_scope,
            address_diagnostic=address_diagnostic,
            screening=screening,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic closed-log / CurveFP experiments.")
    parser.add_argument("--traces", type=Path, default=Path("results/closed_log/traces"), help="operand-trace directory")
    parser.add_argument("--output", type=Path, default=Path("results/closed_log/experiments"), help="CSV/JSON output directory")
    parser.add_argument("--sections", type=_parse_sections, default={"baseline", "radix", "cheap", "group", "layout"})
    parser.add_argument("--capture", action="store_true", help="regenerate traces; otherwise reuse an existing manifest")
    parser.add_argument("--no-real", action="store_true", help="with --capture, generate retained synthetic traces only")
    parser.add_argument("--local-files-only", action="store_true", help="with --capture, forbid pretrained-model download")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline-group-size", type=int, default=64)
    parser.add_argument("--group-sizes", type=_parse_csv_ints, default=DEFAULT_GROUP_SIZES)
    parser.add_argument("--total-bits", type=_parse_csv_ints, default=(7, 8))
    parser.add_argument("--continuous-points", type=int, default=31)
    parser.add_argument("--rational-max-q", type=int, default=16)
    parser.add_argument(
        "--product-candidate-limit",
        type=int,
        default=0,
        help="0 = every q-bounded candidate; positive = explicit evenly-spaced landmark subset",
    )
    parser.add_argument("--product-trace-limit", type=int, default=0, help="0 = every stored trace")
    parser.add_argument("--trace-limit", type=int, default=0, help="0 = every stored trace for every selected phase")
    parser.add_argument("--scalar-sample-values", type=int, default=8192)
    parser.add_argument("--synthetic-scalar-count", type=int, default=20_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="explicit small deterministic screen: no model capture, q<=6, 11 continuous bases, 16 product candidates, four traces",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rational_max_q = args.rational_max_q
    continuous_points = args.continuous_points
    product_candidate_limit = args.product_candidate_limit
    product_trace_limit = args.product_trace_limit
    trace_limit = args.trace_limit
    scalar_sample_values = args.scalar_sample_values
    synthetic_scalar_count = args.synthetic_scalar_count
    if args.quick:
        rational_max_q = min(rational_max_q, 6)
        continuous_points = min(continuous_points, 11)
        product_candidate_limit = 16 if product_candidate_limit == 0 else min(product_candidate_limit, 16)
        product_trace_limit = 4 if product_trace_limit == 0 else min(product_trace_limit, 4)
        trace_limit = 4 if trace_limit == 0 else min(trace_limit, 4)
        scalar_sample_values = min(scalar_sample_values, 2048)
        synthetic_scalar_count = min(synthetic_scalar_count, 4096)
    run_experiments(
        trace_dir=args.traces,
        output=args.output,
        sections=set(args.sections),
        capture=args.capture,
        include_real=not args.no_real,
        local_files_only=args.local_files_only,
        seed=args.seed,
        baseline_group_size=args.baseline_group_size,
        group_sizes=args.group_sizes,
        total_bits=args.total_bits,
        continuous_points=continuous_points,
        rational_max_q=rational_max_q,
        product_candidate_limit=product_candidate_limit,
        product_trace_limit=product_trace_limit,
        trace_limit=trace_limit,
        scalar_sample_values=scalar_sample_values,
        synthetic_scalar_count=synthetic_scalar_count,
    )


if __name__ == "__main__":
    main()
