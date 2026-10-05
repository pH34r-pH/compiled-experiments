"""Scalar representation baselines for factor-algebra feasibility experiments.

All encoders are deterministic NumPy reference implementations.  They measure
storage faithfully, but make no claim about kernel cost or addition behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


NEAR_ZERO = 1e-8


def bits_for_unsigned(max_value: int) -> int:
    """Bits needed for the inclusive integer range 0..max_value."""
    return max(1, math.ceil(math.log2(max_value + 1)))


def relative_error(reference: np.ndarray, reconstructed: np.ndarray, floor: float = NEAR_ZERO) -> np.ndarray:
    """Absolute relative error with a fixed denominator floor around zero."""
    return np.abs(reconstructed - reference) / np.maximum(np.abs(reference), floor)


@dataclass(frozen=True)
class Metrics:
    mean_absolute_error: float
    mean_relative_error: float
    mse: float
    rmse: float
    max_absolute_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mean_absolute_error,
            "mean_relative_error": self.mean_relative_error,
            "mse": self.mse,
            "rmse": self.rmse,
            "max_absolute_error": self.max_absolute_error,
        }


def compute_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> Metrics:
    error = reconstructed - reference
    return Metrics(
        mean_absolute_error=float(np.mean(np.abs(error))),
        mean_relative_error=float(np.mean(relative_error(reference, reconstructed))),
        mse=float(np.mean(error**2)),
        rmse=float(np.sqrt(np.mean(error**2))),
        max_absolute_error=float(np.max(np.abs(error))),
    )


class ScalarEncoder(Protocol):
    name: str

    def encode(self, values: np.ndarray) -> object: ...
    def decode(self, encoded: object) -> np.ndarray: ...
    def bits_per_value(self, count: int) -> float: ...
    def positive_levels(self, low: float, high: float) -> np.ndarray: ...


class FP16Encoder:
    name = "fp16"

    def encode(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float16)

    def decode(self, encoded: object) -> np.ndarray:
        return np.asarray(encoded, dtype=np.float32)

    def bits_per_value(self, count: int) -> float:
        return 16.0

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        # Enumerating all binary16 values is small and makes spacing exact.
        raw = np.arange(1, 0x7C00, dtype=np.uint16)
        values = raw.view(np.float16).astype(np.float64)
        return values[(values >= low) & (values <= high)]


@dataclass(frozen=True)
class Int8Payload:
    values: np.ndarray
    scale: float


class Int8ScaleEncoder:
    name = "int8_scale"

    def encode(self, values: np.ndarray) -> Int8Payload:
        values = np.asarray(values, dtype=np.float32)
        scale = float(np.max(np.abs(values)) / 127.0) if np.any(values) else 1.0
        return Int8Payload(np.clip(np.rint(values / scale), -127, 127).astype(np.int8), scale)

    def decode(self, encoded: object) -> np.ndarray:
        payload = encoded
        assert isinstance(payload, Int8Payload)
        return payload.values.astype(np.float32) * payload.scale

    def bits_per_value(self, count: int) -> float:
        return 8.0 + 32.0 / count  # one FP32 tensor scale, never hidden

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        # Scale is tensor-dependent; normalized levels make that dependency visible.
        return np.arange(1, 128, dtype=np.float64) / 127.0


@dataclass(frozen=True)
class LogPayload:
    codes: np.ndarray


class LogEncoder:
    """Signed powers of two plus exact zero, clipped to configured exponents."""
    name = "power_of_two"

    def __init__(self, min_exponent: int = -16, max_exponent: int = 8):
        if min_exponent > max_exponent:
            raise ValueError("min_exponent must not exceed max_exponent")
        self.min_exponent, self.max_exponent = min_exponent, max_exponent
        self.exponents = np.arange(min_exponent, max_exponent + 1)
        self._magnitudes = np.exp2(self.exponents).astype(np.float64)

    def encode(self, values: np.ndarray) -> LogPayload:
        x = np.asarray(values, dtype=np.float64)
        code = np.zeros(x.shape, dtype=np.int16)
        nonzero = x != 0
        exponent_index = np.clip(
            np.rint(np.log2(np.abs(x[nonzero]))).astype(int) - self.min_exponent,
            0,
            len(self.exponents) - 1,
        )
        # zero=0; positive codes 1..L; negative codes L+1..2L
        code[nonzero] = 1 + exponent_index + (x[nonzero] < 0) * len(self.exponents)
        return LogPayload(code)

    def decode(self, encoded: object) -> np.ndarray:
        payload = encoded
        assert isinstance(payload, LogPayload)
        code = payload.codes
        result = np.zeros(code.shape, dtype=np.float32)
        nonzero = code != 0
        index = (code[nonzero] - 1) % len(self.exponents)
        sign = np.where(code[nonzero] <= len(self.exponents), 1.0, -1.0)
        result[nonzero] = (sign * self._magnitudes[index]).astype(np.float32)
        return result

    def bits_per_value(self, count: int) -> float:
        return float(bits_for_unsigned(2 * len(self.exponents)))

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        return self._magnitudes[(self._magnitudes >= low) & (self._magnitudes <= high)]


@dataclass(frozen=True)
class RationalPayload:
    numerator: np.ndarray
    denominator: np.ndarray


class RationalPairEncoder:
    """Bounded n/d reference encoder; numerator and denominator allocation is explicit."""
    name = "rational_pair"

    def __init__(self, numerator_limit: int = 32767, denominator_limit: int = 4095):
        if numerator_limit < 1 or denominator_limit < 1:
            raise ValueError("rational bounds must be positive")
        self.numerator_limit, self.denominator_limit = numerator_limit, denominator_limit

    def encode(self, values: np.ndarray) -> RationalPayload:
        x = np.asarray(values, dtype=np.float64).ravel()
        numerator = np.empty(x.size, dtype=np.int32)
        denominator = np.empty(x.size, dtype=np.int32)
        for i, value in enumerate(x):
            if value == 0 or not np.isfinite(value):
                numerator[i], denominator[i] = 0, 1
                continue
            fraction = Fraction(float(value)).limit_denominator(self.denominator_limit)
            # Preserve the denominator bound; clamp numerator rather than silently widening it.
            numerator[i] = int(np.clip(fraction.numerator, -self.numerator_limit, self.numerator_limit))
            denominator[i] = fraction.denominator
        return RationalPayload(numerator.reshape(x.shape if np.asarray(values).ndim == 1 else np.asarray(values).shape), denominator.reshape(np.asarray(values).shape))

    def decode(self, encoded: object) -> np.ndarray:
        payload = encoded
        assert isinstance(payload, RationalPayload)
        return payload.numerator.astype(np.float32) / payload.denominator.astype(np.float32)

    def bits_per_value(self, count: int) -> float:
        numerator_bits = bits_for_unsigned(2 * self.numerator_limit)
        denominator_bits = bits_for_unsigned(self.denominator_limit)
        return float(numerator_bits + denominator_bits)

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        # Exact enumeration is intentionally avoided: this representation has O(N*D) levels.
        # A deterministic reduced sample still diagnoses its highly nonuniform spacing.
        denominators = np.arange(1, min(self.denominator_limit, 256) + 1)
        numerators = np.arange(1, min(self.numerator_limit, 4096) + 1)
        values = (numerators[:, None] / denominators).ravel()
        values = np.unique(values[(values >= low) & (values <= high)])
        return values


@dataclass(frozen=True)
class FactorPayload:
    sign: np.ndarray
    exponents: np.ndarray


class FactorBasisEncoder:
    """Nearest fixed-basis value in log space with dense exponent-vector storage."""
    name = "factor_basis_dense"

    def __init__(self, basis: Iterable[int] = (2, 3, 5, 7), exponent_limit: int = 2):
        basis = tuple(int(b) for b in basis)
        if not basis or any(b <= 1 for b in basis) or exponent_limit < 0:
            raise ValueError("basis must contain values >1 and exponent_limit must be nonnegative")
        self.basis, self.exponent_limit = basis, exponent_limit
        mesh = np.array(np.meshgrid(*([np.arange(-exponent_limit, exponent_limit + 1)] * len(basis)), indexing="ij"))
        self._all_exponents = mesh.reshape(len(basis), -1).T.astype(np.int16)
        logs = np.log(np.asarray(basis, dtype=np.float64))
        self._all_log_values = self._all_exponents @ logs
        self._all_values = np.exp(np.clip(self._all_log_values, -700, 700))

    @property
    def exponent_bits(self) -> int:
        return bits_for_unsigned(2 * self.exponent_limit)

    def encode(self, values: np.ndarray) -> FactorPayload:
        x = np.asarray(values, dtype=np.float64)
        flat = x.ravel()
        sign = np.sign(flat).astype(np.int8)
        exponents = np.zeros((flat.size, len(self.basis)), dtype=np.int16)
        nonzero = sign != 0
        if np.any(nonzero):
            target = np.log(np.abs(flat[nonzero]))
            # This is a scalar baseline. Chunking avoids a huge N x codebook allocation.
            choices = np.empty(target.size, dtype=np.intp)
            for start in range(0, target.size, 4096):
                chunk = target[start:start + 4096]
                choices[start:start + len(chunk)] = np.abs(chunk[:, None] - self._all_log_values).argmin(axis=1)
            exponents[nonzero] = self._all_exponents[choices]
        return FactorPayload(sign.reshape(x.shape), exponents.reshape(*x.shape, len(self.basis)))

    def decode(self, encoded: object) -> np.ndarray:
        payload = encoded
        assert isinstance(payload, FactorPayload)
        log_value = payload.exponents.astype(np.float64) @ np.log(np.asarray(self.basis, dtype=np.float64))
        return (payload.sign * np.exp(np.clip(log_value, -700, 700))).astype(np.float32)

    def bits_per_value(self, count: int) -> float:
        # Explicit zero flag + sign bit + every exponent, including zero exponents.
        return float(2 + len(self.basis) * self.exponent_bits)

    def literal_factor_list_bits(self, encoded: FactorPayload) -> float:
        """Actual average bits if only nonzero factors are listed per value."""
        count = encoded.exponents.shape[-1]
        nonzero_factors = np.count_nonzero(encoded.exponents, axis=-1)
        factor_index_bits = bits_for_unsigned(count - 1)
        count_bits = bits_for_unsigned(count)
        per = 2 + count_bits + nonzero_factors * (factor_index_bits + self.exponent_bits)
        return float(np.mean(per))

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        values = np.unique(self._all_values)
        return values[(values >= low) & (values <= high)]


@dataclass(frozen=True)
class CodebookPayload:
    ids: np.ndarray


class FactorCodebookEncoder:
    """IDs into a fixed factor-basis dictionary, with all dictionary cost amortized."""
    name = "factor_codebook"

    def __init__(self, factor_encoder: FactorBasisEncoder, size: int = 32, calibration: np.ndarray | None = None):
        candidates = factor_encoder.positive_levels(0.0, float("inf"))
        if size < 2 or size > len(candidates):
            raise ValueError("codebook size must be in [2, number of factor candidates]")
        if calibration is None or not np.any(np.asarray(calibration) != 0):
            low, high = candidates[0], candidates[-1]
        else:
            magnitude = np.abs(np.asarray(calibration, dtype=np.float64))
            magnitude = magnitude[magnitude > 0]
            low, high = max(candidates[0], float(np.quantile(magnitude, 0.001))), min(candidates[-1], float(np.quantile(magnitude, 0.999)))
        within = candidates[(candidates >= low) & (candidates <= high)]
        positions = np.unique(np.rint(np.linspace(0, len(within) - 1, size)).astype(int))
        positive = within[positions]
        self.factor_encoder, self.size = factor_encoder, len(positive)
        self.entries = np.concatenate(([0.0], positive, -positive)).astype(np.float32)
        self._log_abs = np.log(positive)

    def encode(self, values: np.ndarray) -> CodebookPayload:
        x = np.asarray(values, dtype=np.float64)
        flat = x.ravel()
        ids = np.zeros(flat.size, dtype=np.int32)
        nonzero = flat != 0
        target = np.log(np.abs(flat[nonzero]))
        indices = np.empty(target.size, dtype=np.intp)
        for start in range(0, target.size, 4096):
            chunk = target[start:start + 4096]
            indices[start:start + len(chunk)] = np.abs(chunk[:, None] - self._log_abs).argmin(axis=1)
        ids[nonzero] = 1 + indices + (flat[nonzero] < 0) * self.size
        return CodebookPayload(ids.reshape(x.shape))

    def decode(self, encoded: object) -> np.ndarray:
        payload = encoded
        assert isinstance(payload, CodebookPayload)
        return self.entries[payload.ids]

    def bits_per_value(self, count: int) -> float:
        # IDs + every dictionary entry as a dense factor vector; no free static dictionary.
        return float(bits_for_unsigned(len(self.entries) - 1) + len(self.entries) * self.factor_encoder.bits_per_value(1) / count)

    def positive_levels(self, low: float, high: float) -> np.ndarray:
        values = self.entries[self.entries > 0]
        return values[(values >= low) & (values <= high)]


def synthetic_distributions(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Normal 0.02 is a conventional small-weight proxy. The activation proxy has
    # a core plus 1% outliers, deliberately exposing range-sensitive encoders.
    weights = rng.normal(0.0, 0.02, count).astype(np.float32)
    activations = rng.normal(0.0, 1.0, count).astype(np.float32)
    outlier = rng.random(count) < 0.01
    activations[outlier] += rng.normal(0.0, 6.0, int(outlier.sum())).astype(np.float32)
    return {"weights_normal_0.02": weights, "activations_core_plus_outliers": activations}


def evaluate_encoder(encoder: ScalarEncoder, values: np.ndarray) -> tuple[dict[str, float], np.ndarray, object]:
    encoded = encoder.encode(values)
    reconstructed = encoder.decode(encoded)
    result = compute_metrics(values, reconstructed).as_dict()
    result["bits_per_value"] = encoder.bits_per_value(values.size)
    if isinstance(encoder, FactorBasisEncoder):
        result["literal_factor_list_bits_per_value"] = encoder.literal_factor_list_bits(encoded)
    return result, reconstructed, encoded


def spacing_summary(encoder: ScalarEncoder, low: float, high: float) -> dict[str, np.ndarray]:
    levels = np.unique(encoder.positive_levels(low, high))
    levels = levels[np.isfinite(levels) & (levels > 0)]
    levels.sort()
    if len(levels) < 2:
        return {"level": np.array([]), "log10_spacing": np.array([])}
    return {"level": levels[1:], "log10_spacing": np.diff(np.log10(levels))}


def plot_results(rows: list[dict[str, object]], reconstructions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], encoders: list[ScalarEncoder], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for distribution in sorted({str(r["distribution"]) for r in rows}):
        subset = [r for r in rows if r["distribution"] == distribution]
        ax.scatter([r["bits_per_value"] for r in subset], [r["rmse"] for r in subset], label=distribution)
        for row in subset:
            ax.annotate(str(row["representation"]), (float(row["bits_per_value"]), float(row["rmse"])), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_yscale("log"); ax.set_xlabel("effective bits / value"); ax.set_ylabel("RMSE"); ax.set_title("Reconstruction error versus accounted storage"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "error_vs_bits.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Use the activation distribution; error quantiles survive severe outliers better than a histogram.
    distribution = "activations_core_plus_outliers"
    for encoder in encoders:
        original, reconstructed = reconstructions[(distribution, encoder.name)]
        absolute = np.abs(reconstructed - original)
        quantiles = np.quantile(absolute, np.linspace(0, 1, 501))
        ax.plot(np.linspace(0, 100, len(quantiles)), np.maximum(quantiles, 1e-12), label=encoder.name)
    ax.set_yscale("log"); ax.set_xlabel("absolute-error percentile"); ax.set_ylabel("absolute error"); ax.set_title("Activation reconstruction-error distributions"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "error_distribution.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for encoder in encoders:
        summary = spacing_summary(encoder, 1e-4, 16.0)
        if len(summary["level"]):
            ax.scatter(np.log10(summary["level"]), summary["log10_spacing"], s=7, alpha=0.65, label=encoder.name)
    ax.set_xlabel("log10(|represented value|)"); ax.set_ylabel("gap in log10 magnitude"); ax.set_title("Positive representable-level spacing (1e-4 to 16)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "representable_spacing.png", dpi=170); plt.close(fig)


def factor_sensitivity(values: np.ndarray, output: Path) -> list[dict[str, object]]:
    import matplotlib.pyplot as plt

    configs = [((2, 3), e) for e in (1, 2, 3)] + [((2, 3, 5, 7), e) for e in (1, 2, 3)]
    rows = []
    for basis, exponent_limit in configs:
        encoder = FactorBasisEncoder(basis, exponent_limit)
        metrics, _, _ = evaluate_encoder(encoder, values)
        rows.append({"basis": "x".join(map(str, basis)), "basis_size": len(basis), "exponent_limit": exponent_limit, **metrics})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for basis in sorted({str(r["basis"]) for r in rows}):
        subset = [r for r in rows if r["basis"] == basis]
        ax.plot([r["bits_per_value"] for r in subset], [r["rmse"] for r in subset], marker="o", label=f"basis {basis}")
    ax.set_yscale("log"); ax.set_xlabel("dense fixed-exponent bits / value"); ax.set_ylabel("RMSE"); ax.set_title("Factor-basis sensitivity on activations"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "factor_sensitivity.png", dpi=170); plt.close(fig)
    return rows


def run_experiment(output: Path, *, count: int = 20_000, seed: int = 145) -> dict[str, object]:
    distributions = synthetic_distributions(count, seed)
    factor = FactorBasisEncoder((2, 3, 5, 7), exponent_limit=2)
    calibration = distributions["activations_core_plus_outliers"][: count // 2]
    encoders: list[ScalarEncoder] = [FP16Encoder(), Int8ScaleEncoder(), LogEncoder(), RationalPairEncoder(), factor, FactorCodebookEncoder(factor, 32, calibration)]
    rows: list[dict[str, object]] = []
    reconstructions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for distribution, values in distributions.items():
        for encoder in encoders:
            metrics, reconstructed, _ = evaluate_encoder(encoder, values)
            rows.append({"distribution": distribution, "representation": encoder.name, **metrics})
            reconstructions[(distribution, encoder.name)] = (values, reconstructed)
    sensitivity = factor_sensitivity(distributions["activations_core_plus_outliers"], output)
    plot_results(rows, reconstructions, encoders, output)
    payload = {
        "assumptions": {
            "seed": seed,
            "values_per_distribution": count,
            "relative_error_floor": NEAR_ZERO,
            "factor_basis": [2, 3, 5, 7],
            "factor_exponent_limit": 2,
            "factor_storage": "dense: zero flag + sign bit + every bounded exponent; literal-list cost reported separately",
            "codebook_storage": "ID plus all dictionary entries encoded as dense factor vectors, amortized over one tensor",
            "real_tensor_samples": "not included: repository has no pretrained-tensor loader",
        },
        "results": rows,
        "factor_sensitivity": sensitivity,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    columns = list(rows[0])
    csv = ",".join(columns) + "\n" + "\n".join(",".join(str(row.get(column, "")) for column in columns) for row in rows) + "\n"
    (output / "scalar_results.csv").write_text(csv)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the factor-algebra scalar feasibility benchmark.")
    parser.add_argument("--output", type=Path, default=Path("results/factor_algebra"))
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=145)
    args = parser.parse_args()
    run_experiment(args.output, count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
