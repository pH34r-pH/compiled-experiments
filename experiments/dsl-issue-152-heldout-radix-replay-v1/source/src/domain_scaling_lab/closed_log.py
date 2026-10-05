"""Reference closed-log datatype and CurveFP-v2 product-address semantics.

This module is intentionally a numerical/algebraic reference, not a hardware
claim.  It reproduces the finite rational-radix address algebra described in
CurveFP v2: ``m=0`` is zero, a nonzero rank expands to an exponent/curve
coordinate, products map to a signed ``(phase, binary-exponent)`` counter, and
both a histogram and a per-phase Kulisch-style reduction agree with immediate
materialization.  Bank layout, pipeline placement, and actual cost are left to
the explicitly labelled analysis module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


def counter_width_bits(dot_width: int) -> int:
    """Conservative signed counter width for at most ``dot_width`` ±1 updates."""
    if dot_width < 0:
        raise ValueError("dot_width must be nonnegative")
    return 1 + math.ceil(math.log2(dot_width + 1))


@dataclass(frozen=True)
class ClosedLogSpec:
    """A finite CurveFP-style E/C format with rational radix ``2**(p/q)``.

    ``E`` and ``C`` are exponent and curve fields; total storage is sign + E + C
    bits.  Rank ``m=0`` is the zero sentinel.  The remaining ranks map to
    ``n=m-2**(E-1)*K`` where ``n=e*K+k`` and ``K=2**C``.  This leaves exactly
    one finite coordinate unused for zero rather than pretending it is a free
    nonzero state.
    """

    E: int
    C: int
    p: int = 1
    q: int = 1
    scale_bits: int = 8
    scale_policy: Literal["curvefp_ceil_absmax", "endpoint_cover"] = "curvefp_ceil_absmax"

    def __post_init__(self) -> None:
        if self.E < 1 or self.C < 0:
            raise ValueError("E must be at least one bit and C must be nonnegative")
        if self.p <= 0 or self.q <= 0:
            raise ValueError("rational radix requires positive p and q")
        if self.scale_bits < 1:
            raise ValueError("scale_bits must be positive")
        if self.scale_policy not in {"curvefp_ceil_absmax", "endpoint_cover"}:
            raise ValueError("scale_policy must be 'curvefp_ceil_absmax' or 'endpoint_cover'")
        divisor = math.gcd(self.p, self.q)
        if divisor != 1:
            object.__setattr__(self, "p", self.p // divisor)
            object.__setattr__(self, "q", self.q // divisor)

    @property
    def K(self) -> int:
        return 1 << self.C

    @property
    def total_bits(self) -> int:
        return 1 + self.E + self.C

    @property
    def rank_count(self) -> int:
        return 1 << (self.E + self.C)

    @property
    def nonzero_state_count(self) -> int:
        return self.rank_count - 1

    @property
    def n_offset(self) -> int:
        return (1 << (self.E - 1)) * self.K

    @property
    def n_min(self) -> int:
        # m=0 is zero sentinel, so the first representable nonzero rank is m=1.
        return 1 - self.n_offset

    @property
    def n_max(self) -> int:
        return self.rank_count - 1 - self.n_offset

    @property
    def alpha(self) -> float:
        """``log2(radix)``."""
        return self.p / self.q

    @property
    def radix(self) -> float:
        return float(math.exp2(self.alpha))

    @property
    def H(self) -> int:
        """Exact finite phase count, CurveFP v2 Eq. 4."""
        return self.q * self.K // math.gcd(self.p, self.q * self.K)

    @property
    def n_values(self) -> np.ndarray:
        return np.arange(self.n_min, self.n_max + 1, dtype=np.int32)

    @property
    def local_levels(self) -> np.ndarray:
        return np.exp2(self.alpha * self.n_values.astype(np.float64) / self.K)

    def rank_to_n(self, rank: np.ndarray | int) -> np.ndarray:
        rank_array = np.asarray(rank, dtype=np.int64)
        if np.any(rank_array < 0) or np.any(rank_array >= self.rank_count):
            raise ValueError("rank outside format range")
        return rank_array - self.n_offset

    def n_to_rank(self, n: np.ndarray | int) -> np.ndarray:
        n_array = np.asarray(n, dtype=np.int64)
        if np.any(n_array < self.n_min) or np.any(n_array > self.n_max):
            raise ValueError("n outside representable nonzero range")
        return n_array + self.n_offset

    def n_to_fields(self, n: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        n_array = np.asarray(n, dtype=np.int64)
        return np.floor_divide(n_array, self.K), np.mod(n_array, self.K)

    def metadata(self) -> dict[str, object]:
        return {
            "E": self.E,
            "C": self.C,
            "total_bits": self.total_bits,
            "K": self.K,
            "p": self.p,
            "q": self.q,
            "scale_bits": self.scale_bits,
            "radix": self.radix,
            "phase_count_H": self.H,
            "nonzero_states": self.nonzero_state_count,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "zero_sentinel": "rank m=0",
            "scale_policy": self.scale_policy,
            "curvefp_version": "arXiv:2608.10010v2 (consulted 2026-09-01)",
        }

    # Convenient instance aliases keep callers from duplicating API plumbing.
    def quantize(self, values: np.ndarray, group_size: int | None = None, axis: int = -1) -> "ClosedLogPayload":
        return quantize_grouped(values, self, group_size=group_size, axis=axis)

    def decode(self, payload: "ClosedLogPayload") -> np.ndarray:
        return decode_grouped(payload)

    def product_descriptors(self, lhs: "ClosedLogPayload", rhs: "ClosedLogPayload") -> "ProductDescriptors":
        return product_descriptors(lhs, rhs)


@dataclass(frozen=True)
class ClosedLogPayload:
    spec: ClosedLogSpec
    sign: np.ndarray
    rank: np.ndarray
    scale_exponent: np.ndarray
    group_size: int | None
    axis: int
    clipped: np.ndarray
    scale_clipped: np.ndarray

    @property
    def nonzero(self) -> np.ndarray:
        return self.rank != 0

    @property
    def n(self) -> np.ndarray:
        # The sentinel's n is arbitrary and always masked by ``nonzero``.
        result = self.spec.rank_to_n(np.clip(self.rank, 0, self.spec.rank_count - 1))
        return result.astype(np.int32)

    @property
    def exponent(self) -> np.ndarray:
        exponent, _ = self.spec.n_to_fields(self.n)
        return exponent.astype(np.int32)

    @property
    def curve(self) -> np.ndarray:
        _, curve = self.spec.n_to_fields(self.n)
        return curve.astype(np.int32)

    def expanded_scale_exponent(self) -> np.ndarray:
        """Broadcast group scale exponents back to the element tensor shape."""
        axis = self.axis if self.axis >= 0 else self.axis + self.rank.ndim
        moved_shape = list(np.moveaxis(self.rank, axis, -1).shape)
        width = moved_shape[-1]
        if self.group_size is None:
            # A tensorwise scale is stored as one scalar / rank-ndim prefix.
            expanded = np.full(moved_shape, int(np.asarray(self.scale_exponent).reshape(-1)[0]), dtype=np.int32)
        else:
            group = self.group_size
            positions = np.arange(width, dtype=np.int64) // group
            expanded = np.take(self.scale_exponent, positions, axis=-1).astype(np.int32)
        return np.moveaxis(expanded, -1, axis)


def _nearest_n_absolute(normalized_magnitude: np.ndarray, *, spec: ClosedLogSpec) -> np.ndarray:
    """Select the nearest reconstructed local magnitude in absolute error.

    CurveFP's quantizer is specified as nearest reconstructed magnitude, not
    nearest coordinate in log space.  Those rules differ at a geometric mean;
    this helper intentionally uses the arithmetic midpoint and deterministic
    lower-level tie break.
    """
    levels = spec.local_levels
    right = np.searchsorted(levels, normalized_magnitude, side="left")
    right = np.clip(right, 0, len(levels) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(levels[right] - normalized_magnitude) < np.abs(levels[left] - normalized_magnitude)
    index = np.where(choose_right, right, left)
    return (index + spec.n_min).astype(np.int32)


def quantize_grouped(
    values: np.ndarray,
    spec: ClosedLogSpec,
    group_size: int | None = None,
    axis: int = -1,
) -> ClosedLogPayload:
    """Quantize with power-of-two block scales along ``axis``.

    ``curvefp_ceil_absmax`` is CurveFP v2's stated dynamic scale policy,
    ``ceil(log2(max(abs(block))))``.  ``endpoint_cover`` shifts the scale to
    make the top local level cover the maximum; that matches #148's normalized
    legacy scalar grid but is explicitly an engineering variation, not a
    faithful direct reproduction.  ``group_size=None`` is one tensorwise scale;
    an integer group size creates independent contiguous groups for every
    prefix row along ``axis``.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("values must have at least one dimension")
    normalized_axis = axis if axis >= 0 else x.ndim + axis
    if normalized_axis < 0 or normalized_axis >= x.ndim:
        raise ValueError("axis out of range")
    if np.any(~np.isfinite(x)):
        raise ValueError("closed-log reference requires finite values")
    moved = np.moveaxis(x, normalized_axis, -1)
    width = moved.shape[-1]
    if group_size is not None and (group_size < 1 or group_size > width):
        raise ValueError("group_size must be in [1, axis length], or None")
    sign_moved = np.sign(moved).astype(np.int8)
    rank_moved = np.zeros(moved.shape, dtype=np.int32)
    clipped_moved = np.zeros(moved.shape, dtype=bool)
    max_log2_local = spec.alpha * spec.n_max / spec.K
    scale_min = -(1 << (spec.scale_bits - 1))
    scale_max = (1 << (spec.scale_bits - 1)) - 1

    if group_size is None:
        magnitude = np.abs(moved)
        maximum = float(magnitude.max())
        if maximum > 0:
            requested_scale = int(math.ceil(math.log2(maximum)))
            if spec.scale_policy == "endpoint_cover":
                requested_scale = int(math.ceil(math.log2(maximum) - max_log2_local))
        else:
            requested_scale = 0
        scale = int(np.clip(requested_scale, scale_min, scale_max))
        positive = magnitude > 0
        n = _nearest_n_absolute(magnitude / math.exp2(scale), spec=spec)
        rank_moved[positive] = spec.n_to_rank(n[positive]).astype(np.int32)
        normalized = magnitude / math.exp2(scale)
        clipped_moved[positive] = (normalized[positive] < spec.local_levels[0]) | (normalized[positive] > spec.local_levels[-1])
        scale_exponent = np.asarray([scale], dtype=np.int32)
        scale_clipped = np.asarray([requested_scale != scale], dtype=bool)
    else:
        prefix = moved.shape[:-1]
        group_count = math.ceil(width / group_size)
        scale_exponent = np.zeros(prefix + (group_count,), dtype=np.int32)
        scale_clipped = np.zeros(prefix + (group_count,), dtype=bool)
        for group_index, start in enumerate(range(0, width, group_size)):
            stop = min(width, start + group_size)
            block = moved[..., start:stop]
            magnitude = np.abs(block)
            maximum = np.max(magnitude, axis=-1)
            requested_scale = np.where(maximum > 0, np.ceil(np.log2(np.maximum(maximum, np.finfo(np.float64).tiny))), 0).astype(np.int32)
            if spec.scale_policy == "endpoint_cover":
                requested_scale = np.where(maximum > 0, np.ceil(np.log2(np.maximum(maximum, np.finfo(np.float64).tiny)) - max_log2_local), 0).astype(np.int32)
            scale = np.clip(requested_scale, scale_min, scale_max).astype(np.int32)
            scale_exponent[..., group_index] = scale
            scale_clipped[..., group_index] = requested_scale != scale
            positive = magnitude > 0
            normalized = magnitude / np.exp2(scale[..., None])
            n = _nearest_n_absolute(normalized, spec=spec)
            rank_block = rank_moved[..., start:stop]
            rank_block[positive] = spec.n_to_rank(n[positive]).astype(np.int32)
            rank_moved[..., start:stop] = rank_block
            clipped_moved[..., start:stop] = positive & ((normalized < spec.local_levels[0]) | (normalized > spec.local_levels[-1]))
    return ClosedLogPayload(
        spec=spec,
        sign=np.moveaxis(sign_moved, -1, normalized_axis),
        rank=np.moveaxis(rank_moved, -1, normalized_axis),
        scale_exponent=scale_exponent,
        group_size=group_size,
        axis=normalized_axis,
        clipped=np.moveaxis(clipped_moved, -1, normalized_axis),
        scale_clipped=scale_clipped,
    )


def decode_grouped(payload: ClosedLogPayload) -> np.ndarray:
    n = payload.n.astype(np.float64)
    scale = payload.expanded_scale_exponent().astype(np.float64)
    result = payload.sign.astype(np.float64) * np.exp2(scale + payload.spec.alpha * n / payload.spec.K)
    result[~payload.nonzero] = 0.0
    return result.astype(np.float32)


@dataclass(frozen=True)
class ProductDescriptors:
    spec: ClosedLogSpec
    sign: np.ndarray
    phase: np.ndarray
    exponent: np.ndarray
    address: np.ndarray
    n_lhs: np.ndarray
    n_rhs: np.ndarray
    u: np.ndarray
    lhs_scale_exponent: np.ndarray
    rhs_scale_exponent: np.ndarray
    nonzero: np.ndarray


def product_descriptors(lhs: ClosedLogPayload, rhs: ClosedLogPayload) -> ProductDescriptors:
    """Create exact CurveFP-style product descriptors before destination bounds."""
    if lhs.spec != rhs.spec:
        raise ValueError("operands need identical closed-log specs for this reference path")
    if lhs.rank.shape != rhs.rank.shape:
        raise ValueError("operands need matching shapes")
    spec = lhs.spec
    n_lhs, n_rhs = lhs.n, rhs.n
    u = spec.p * (n_lhs.astype(np.int64) + n_rhs.astype(np.int64))
    denominator = spec.q * spec.K
    divisor = math.gcd(spec.p, denominator)
    phase = (np.mod(u, denominator) // divisor).astype(np.int32)
    exponent = (lhs.expanded_scale_exponent().astype(np.int64) + rhs.expanded_scale_exponent().astype(np.int64) + np.floor_divide(u, denominator)).astype(np.int32)
    nonzero = lhs.nonzero & rhs.nonzero
    sign = (lhs.sign.astype(np.int8) * rhs.sign.astype(np.int8)).astype(np.int8)
    sign[~nonzero] = 0
    # Last dimension is suitable for ``np.unique`` or routing-key serialization.
    address = np.stack((phase, exponent), axis=-1)
    return ProductDescriptors(
        spec=spec,
        sign=sign,
        phase=phase,
        exponent=exponent,
        address=address,
        n_lhs=n_lhs.astype(np.int32),
        n_rhs=n_rhs.astype(np.int32),
        u=u.astype(np.int64),
        lhs_scale_exponent=lhs.expanded_scale_exponent().astype(np.int32),
        rhs_scale_exponent=rhs.expanded_scale_exponent().astype(np.int32),
        nonzero=nonzero,
    )


@dataclass(frozen=True)
class AccumulationResult:
    value: np.ndarray
    bookkeeping: dict[str, object]


def _descriptor_rows(descriptors: ProductDescriptors) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if descriptors.phase.ndim != 2:
        raise ValueError("accumulation reference expects [dot,width] descriptors")
    return descriptors.phase, descriptors.exponent, descriptors.sign, descriptors.nonzero


def immediate_accumulate(descriptors: ProductDescriptors) -> AccumulationResult:
    phase, exponent, sign, nonzero = _descriptor_rows(descriptors)
    values = sign.astype(np.float64) * np.exp2(exponent.astype(np.float64) + phase.astype(np.float64) / descriptors.spec.H)
    values[~nonzero] = 0.0
    return AccumulationResult(
        value=np.sum(values, axis=1),
        bookkeeping={
            "product_materializations": int(nonzero.sum()),
            "floating_adds_proxy": int(nonzero.sum()),
            "classification": "immediate_materialization_baseline",
        },
    )


def signed_histogram_accumulate(descriptors: ProductDescriptors) -> AccumulationResult:
    """CurveFP's exact signed-count histogram semantics (reference form)."""
    phase, exponent, sign, nonzero = _descriptor_rows(descriptors)
    output = np.zeros(phase.shape[0], dtype=np.float64)
    occupied = 0
    nonzero_signed_bins = 0
    for row in range(phase.shape[0]):
        mask = nonzero[row]
        if not np.any(mask):
            continue
        keys = np.empty(int(mask.sum()), dtype=[("h", np.int32), ("t", np.int32)])
        keys["h"], keys["t"] = phase[row, mask], exponent[row, mask]
        unique, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(inverse, weights=sign[row, mask].astype(np.float64), minlength=len(unique))
        output[row] = np.sum(counts * np.exp2(unique["t"].astype(np.float64) + unique["h"].astype(np.float64) / descriptors.spec.H))
        occupied += len(unique)
        nonzero_signed_bins += int(np.count_nonzero(counts))
    return AccumulationResult(
        value=output,
        bookkeeping={
            "signed_counter_updates": int(nonzero.sum()),
            "visited_histogram_addresses": occupied,
            "nonzero_signed_histogram_bins": nonzero_signed_bins,
            "final_materializations": nonzero_signed_bins,
            "counter_width_bits": counter_width_bits(phase.shape[1]),
            "classification": "curvefp_histogram_reproduction",
        },
    )


def kulisch_accumulate(descriptors: ProductDescriptors) -> AccumulationResult:
    """Per-phase signed-shift Kulisch alternative described by CurveFP."""
    phase, exponent, sign, nonzero = _descriptor_rows(descriptors)
    output = np.zeros(phase.shape[0], dtype=np.float64)
    active_phase_materializations = 0
    nonzero_phase_reductions = 0
    max_span = 0
    for row in range(phase.shape[0]):
        mask = nonzero[row]
        if not np.any(mask):
            continue
        t = exponent[row, mask]
        h = phase[row, mask]
        s = sign[row, mask]
        t_min, t_max = int(t.min()), int(t.max())
        max_span = max(max_span, t_max - t_min + 1)
        value = 0.0
        for current_phase in range(descriptors.spec.H):
            in_phase = h == current_phase
            if not np.any(in_phase):
                continue
            integer = sum(int(delta) << int(exp - t_min) for delta, exp in zip(s[in_phase], t[in_phase], strict=True))
            value += integer * math.exp2(t_min + current_phase / descriptors.spec.H)
            active_phase_materializations += 1
            nonzero_phase_reductions += int(integer != 0)
        output[row] = value
    width = counter_width_bits(phase.shape[1]) + max_span
    return AccumulationResult(
        value=output,
        bookkeeping={
            "signed_shift_updates": int(nonzero.sum()),
            # An active phase is a conservative implementation proxy: it may
            # still cancel to zero before its final coefficient/reduction.
            "active_phase_materializations_conservative": active_phase_materializations,
            "nonzero_phase_reductions": nonzero_phase_reductions,
            "kulisch_width_proxy_bits": width,
            "classification": "curvefp_kulisch_reproduction",
        },
    )


def accumulator_bookkeeping(descriptors: ProductDescriptors) -> dict[str, object]:
    """Return transparent width/count requirements for a descriptor tile."""
    phase, exponent, _, nonzero = _descriptor_rows(descriptors)
    active_exponents = exponent[nonzero]
    span = int(active_exponents.max() - active_exponents.min() + 1) if len(active_exponents) else 0
    return {
        "dot_width": phase.shape[1],
        "phase_count": descriptors.spec.H,
        "signed_counter_width_bits": counter_width_bits(phase.shape[1]),
        "observed_exponent_span": span,
        "kulisch_phase_width_proxy_bits": counter_width_bits(phase.shape[1]) + span,
        "addressing": "(phase, binary exponent); phase histogram and Kulisch are CurveFP v2 prior art",
    }


@dataclass(frozen=True)
class ArbitraryRadixGrid:
    """Scalar-only arbitrary-radix grid used by #155's coordinate control."""

    radix: float
    min_index: int
    max_index: int

    def __post_init__(self) -> None:
        if not self.radix > 1.0 or not math.isfinite(self.radix):
            raise ValueError("radix must be finite and > 1")
        if self.min_index > self.max_index:
            raise ValueError("min_index must be <= max_index")

    @property
    def levels(self) -> np.ndarray:
        return np.exp(np.arange(self.min_index, self.max_index + 1, dtype=np.float64) * math.log(self.radix))

    def log_coordinates(self, values: np.ndarray, log_base: Literal["ln", "log2"] = "ln") -> np.ndarray:
        magnitude = np.abs(np.asarray(values, dtype=np.float64))
        result = np.full(magnitude.shape, -np.inf, dtype=np.float64)
        nonzero = magnitude > 0
        if log_base == "ln":
            result[nonzero] = np.log(magnitude[nonzero]) / math.log(self.radix)
        elif log_base == "log2":
            result[nonzero] = np.log2(magnitude[nonzero]) / math.log2(self.radix)
        else:
            raise ValueError("log_base must be 'ln' or 'log2'")
        return result

    def nearest_indices(self, values: np.ndarray, log_base: Literal["ln", "log2"] = "ln") -> np.ndarray:
        coordinate = self.log_coordinates(values, log_base=log_base)
        result = np.clip(np.rint(coordinate), self.min_index, self.max_index).astype(np.int32)
        # This helper exposes exponent-like coordinate values, so reserve one
        # explicitly out-of-range value for exact zero rather than saturating it
        # to the smallest positive grid level.
        result[np.asarray(values) == 0] = self.min_index - 1
        return result
