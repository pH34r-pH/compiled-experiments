"""Deterministic operand traces for the closed-log / CurveFP experiments.

The trace format intentionally stores small, shape-preserving operand windows,
not a flattened sample of values and not an unbounded stream of products.  A
later experiment can therefore re-quantize the *same* dot products for a new
radix, E/C layout, or scale group without running a model again.

Raw pretrained weights are never downloaded into the repository by this module.
The optional model loader uses an exact revision and saves only selected operand
windows plus their provenance.  Generated ``.npz`` shards are deliberately
small enough for local reuse; a reproducible command and compact summaries are
the commit-friendly deliverables when shards should stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .closed_log import ClosedLogSpec
from .factor_algebra import synthetic_distributions


TRACE_FORMAT_VERSION = 1
PRODUCT_ADDRESS_TRACE_FORMAT_VERSION = 1
DEFAULT_PROMPTS = (
    "Closed logarithmic arithmetic should be evaluated on real transformer GEMMs.",
    "A reusable trace preserves operand topology while allowing radix sweeps.",
    "Block scales separate local quantization density from coarse dynamic range.",
    "Accumulation cost depends on product-address occupancy and locality.",
)
DEFAULT_MODEL_SPECS = (
    # The #148 model, pinned rather than following the mutable ``main`` revision.
    ("google/bert_uncased_L-2_H-128_A-2", "30b0a37ccaaa32f332884b96992754e246e48c5f"),
    # A second independently trained small model.  Capture records a clear
    # unavailable status rather than silently replacing it when it is not cached.
    ("prajjwal1/bert-tiny", "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"),
)
# Some older ``prajjwal1/bert-*`` snapshots contain only model files and a
# minimal legacy BERT config, rather than a tokenizer bundle.  A caller may
# elect to use this *pinned* compatible WordPiece tokenizer; it is never used
# implicitly, so a capture manifest always makes the tokenizer/model pairing
# auditable.
DEFAULT_BERT_FALLBACK_TOKENIZER_SPEC = (
    "google/bert_uncased_L-2_H-128_A-2",
    "30b0a37ccaaa32f332884b96992754e246e48c5f",
)


@dataclass(frozen=True)
class OperandTrace:
    """A deterministic collection of true scalar products from one linear GEMM."""

    trace_id: str
    source: str
    a: np.ndarray
    b: np.ndarray
    reference: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2 or self.a.shape != self.b.shape:
            raise ValueError("a and b must have matching [dot, contracted-width] shapes")
        if self.reference.shape != (self.a.shape[0],):
            raise ValueError("reference must contain one dot product per trace row")

    @property
    def dot_count(self) -> int:
        return int(self.a.shape[0])

    @property
    def width(self) -> int:
        return int(self.a.shape[1])


@dataclass(frozen=True)
class ProductAddressTrace:
    """Compact, directly reusable CurveFP-style product-address shard.

    The shard deliberately stores product descriptors rather than decoded
    values.  ``phase`` plus ``exponent`` is the final accumulator address;
    block-scale context is retained independently so later analysis does not
    need to rerun a model or infer address provenance from values.
    """

    trace_id: str
    representation: str
    spec: ClosedLogSpec
    sign: np.ndarray
    phase: np.ndarray
    exponent: np.ndarray
    nonzero: np.ndarray
    lhs_scale_exponent: np.ndarray
    rhs_scale_exponent: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def address(self) -> np.ndarray:
        return np.stack((self.phase, self.exponent), axis=-1)


def save_product_address_trace(
    directory: Path,
    *,
    trace: OperandTrace,
    representation: str,
    descriptors: object,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Persist one compact direct-address shard and return its manifest row.

    ``descriptors`` is duck-typed so this storage module stays independent of
    the numerical reference implementation.  Its required fields are exactly
    the reusable trace contract, including both scale contexts.
    """
    required = ("sign", "phase", "exponent", "nonzero", "lhs_scale_exponent", "rhs_scale_exponent", "spec")
    missing = [name for name in required if not hasattr(descriptors, name)]
    if missing:
        raise ValueError(f"product descriptors missing required fields: {missing}")
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_id(representation)}--{_safe_id(trace.trace_id)}.npz"
    target = directory / filename
    np.savez_compressed(
        target,
        sign=np.asarray(descriptors.sign, dtype=np.int8),
        phase=np.asarray(descriptors.phase, dtype=np.int16),
        product_exponent=np.asarray(descriptors.exponent, dtype=np.int16),
        nonzero=np.asarray(descriptors.nonzero, dtype=np.bool_),
        lhs_scale_exponent=np.asarray(descriptors.lhs_scale_exponent, dtype=np.int16),
        rhs_scale_exponent=np.asarray(descriptors.rhs_scale_exponent, dtype=np.int16),
    )
    spec = descriptors.spec
    return {
        "trace_id": trace.trace_id,
        "representation": representation,
        "file": filename,
        "source": trace.source,
        "shape": [trace.dot_count, trace.width],
        "descriptor_fields": [
            "sign",
            "phase",
            "product_exponent",
            "(phase, product_exponent) final accumulator address",
            "lhs_scale_exponent",
            "rhs_scale_exponent",
            "nonzero",
        ],
        "spec": spec.metadata(),
        "trace_metadata": trace.metadata,
        **metadata,
        "sha256": _sha256_file(target),
    }


def write_product_address_manifest(directory: Path, entries: Iterable[dict[str, object]], *, generation: dict[str, object]) -> Path:
    """Write the deterministic index consumed by later address analyses."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": PRODUCT_ADDRESS_TRACE_FORMAT_VERSION,
        "kind": "closed_log_product_address_traces",
        "generation": generation,
        "entries": sorted(entries, key=lambda item: (str(item["representation"]), str(item["trace_id"]))),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_product_address_traces(directory: Path, *, verify_checksums: bool = True) -> list[ProductAddressTrace]:
    """Load direct-address shards, checking recorded hashes by default.

    Legacy version-1 entries without ``sha256`` remain readable but unverified.
    As with operand traces, checksum verification may be explicitly disabled
    for forensic/recovery workflows. Path and shape checks always apply.
    """
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != PRODUCT_ADDRESS_TRACE_FORMAT_VERSION:
        raise ValueError(f"unsupported product-address trace format: {manifest.get('format_version')!r}")
    traces: list[ProductAddressTrace] = []
    for entry in manifest.get("entries", []):
        shard_path = (directory / str(entry["file"])).resolve()
        if not shard_path.is_relative_to(directory.resolve()):
            raise ValueError(f"product-address shard path escapes trace directory: {entry['file']!r}")
        if verify_checksums:
            _verify_shard_checksum(shard_path, entry)
        spec_values = dict(entry["spec"])
        spec = ClosedLogSpec(
            E=int(spec_values["E"]),
            C=int(spec_values["C"]),
            p=int(spec_values["p"]),
            q=int(spec_values["q"]),
            scale_bits=int(spec_values.get("scale_bits", 8)),
            scale_policy=str(spec_values["scale_policy"]),
        )
        with np.load(shard_path, allow_pickle=False) as shard:
            expected_shape = tuple(entry["shape"])
            for name in ("sign", "phase", "product_exponent", "nonzero", "lhs_scale_exponent", "rhs_scale_exponent"):
                if len(expected_shape) != 2 or shard[name].shape != expected_shape:
                    raise ValueError(
                        f"product-address shard shape mismatch for {name!r}: "
                        f"expected {expected_shape}, got {shard[name].shape}"
                    )
            traces.append(
                ProductAddressTrace(
                    trace_id=str(entry["trace_id"]),
                    representation=str(entry["representation"]),
                    spec=spec,
                    sign=np.asarray(shard["sign"], dtype=np.int8),
                    phase=np.asarray(shard["phase"], dtype=np.int32),
                    exponent=np.asarray(shard["product_exponent"], dtype=np.int32),
                    nonzero=np.asarray(shard["nonzero"], dtype=bool),
                    lhs_scale_exponent=np.asarray(shard["lhs_scale_exponent"], dtype=np.int32),
                    rhs_scale_exponent=np.asarray(shard["rhs_scale_exponent"], dtype=np.int32),
                    metadata={key: value for key, value in entry.items() if key not in {"file", "spec"}},
                )
            )
    return traces


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _sha256_file(path: Path) -> str:
    """Hash a persisted shard without retaining the complete file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_shard_checksum(shard_path: Path, entry: dict[str, object]) -> None:
    expected_digest = entry.get("sha256")
    if expected_digest is not None:
        actual_digest = _sha256_file(shard_path)
        if actual_digest.lower() != str(expected_digest).lower():
            raise ValueError(
                f"trace shard checksum mismatch for {entry.get('trace_id', shard_path.name)!r}: "
                f"expected {expected_digest}, got {actual_digest}"
            )


def _sha256_json(value: object) -> str:
    """Hash JSON-serializable provenance with a stable encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_evenly(count: int, take: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("cannot select from an empty dimension")
    return np.unique(np.linspace(0, count - 1, min(count, take), dtype=np.int64))


def synthetic_operand_traces(
    widths: Iterable[int] = (64, 128, 256, 512, 1024, 4096),
    *,
    dots_per_width: int = 64,
    seed: int = 146,
) -> list[OperandTrace]:
    """Build retained-distribution A×W dot products at every requested width.

    This is the only source used for the full 4096-width sweep when compact
    pretrained models lack such an actual contracted dimension.  It is labelled
    synthetic in all manifests and summaries.
    """
    distributions = synthetic_distributions(max(20_000, max(widths) * 4), 145)
    activations = distributions["activations_core_plus_outliers"]
    weights = distributions["weights_normal_0.02"]
    traces: list[OperandTrace] = []
    for width in widths:
        if width < 1:
            raise ValueError("trace widths must be positive")
        rng = np.random.default_rng(_stable_seed(seed, "synthetic", width))
        a = rng.choice(activations, size=(dots_per_width, width), replace=True).astype(np.float32)
        b = rng.choice(weights, size=(dots_per_width, width), replace=True).astype(np.float32)
        reference = np.einsum("ij,ij->i", a, b, dtype=np.float64).astype(np.float32)
        traces.append(
            OperandTrace(
                trace_id=f"synthetic-activation-weight-w{width}",
                source="synthetic",
                a=a,
                b=b,
                reference=reference,
                metadata={
                    "synthetic_seed": 145,
                    "sampling_seed": seed,
                    "activation_distribution": "activations_core_plus_outliers",
                    "weight_distribution": "weights_normal_0.02",
                    "width": width,
                    "dots": dots_per_width,
                },
            )
        )
    return traces


def _linear_module_names(model: object) -> list[str]:
    """Choose a compact, role-diverse set of BERT-style linear operations."""
    import torch

    available = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    candidates: list[tuple[str, object]] = []
    # Keep early and late layer instances of four typical transformer roles.
    layer_numbers = [int(m.group(1)) for name, _ in available if (m := re.search(r"(?:encoder|transformer)\.layer\.(\d+)", name))]
    wanted_layers = {min(layer_numbers), max(layer_numbers)} if layer_numbers else set()
    suffixes = (
        "attention.self.query",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    )
    for name, module in available:
        match = re.search(r"(?:encoder|transformer)\.layer\.(\d+)", name)
        layer = int(match.group(1)) if match else None
        if layer in wanted_layers and name.endswith(suffixes):
            candidates.append((name, module))
    # A model with non-standard names still gets reproducible coverage.
    if not candidates:
        candidates = available[: min(8, len(available))]
    return [name for name, _ in candidates]


def _select_valid_token_rows(
    activation: np.ndarray,
    attention_mask: np.ndarray,
    *,
    rows_per_module: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic rows from non-padding BERT token positions.

    The returned coordinates are deliberately retained even though the trace
    stores only the selected operand vectors.  A flat row number alone is
    ambiguous once a prompt batch contains padding, while ``(batch, token)``
    lets a later audit verify that no padding activation contributed a dot.

    Returns ``(selected_rows, valid_coordinates, selected_coordinates,
    selected_flat_indices, selected_indices_within_valid_rows)``.
    """
    activation = np.asarray(activation)
    mask = np.asarray(attention_mask, dtype=bool)
    if rows_per_module < 1:
        raise ValueError("rows_per_module must be positive")
    if activation.ndim != 3:
        raise ValueError(
            "valid-token trace capture requires a [batch, sequence, features] activation; "
            f"got shape {list(activation.shape)}"
        )
    if mask.shape != activation.shape[:2]:
        raise ValueError(
            "attention mask must match the activation [batch, sequence] axes; "
            f"got mask {list(mask.shape)} for activation {list(activation.shape)}"
        )
    valid_coordinates = np.argwhere(mask)
    if len(valid_coordinates) == 0:
        raise ValueError("cannot capture a trace when every token position is padding")
    sequence_length = activation.shape[1]
    valid_flat_indices = valid_coordinates[:, 0] * sequence_length + valid_coordinates[:, 1]
    valid_rows = activation.reshape(-1, activation.shape[-1])[valid_flat_indices]
    selected_indices = _select_evenly(len(valid_rows), rows_per_module)
    selected_coordinates = valid_coordinates[selected_indices]
    selected_flat_indices = valid_flat_indices[selected_indices]
    return (
        valid_rows[selected_indices],
        valid_coordinates,
        selected_coordinates,
        selected_flat_indices,
        selected_indices,
    )


def _load_tokenizer_with_optional_fallback(
    auto_tokenizer: object,
    *,
    model_name: str,
    revision: str,
    fallback_tokenizer_spec: tuple[str, str] | None,
    local_files_only: bool,
) -> tuple[object, dict[str, object]]:
    """Load the model tokenizer, or an explicitly requested pinned fallback."""
    try:
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, revision=revision, use_fast=False, local_files_only=local_files_only
        )
        return tokenizer, {
            "tokenizer_model": model_name,
            "tokenizer_revision": revision,
            "tokenizer_source": "model_snapshot",
        }
    except Exception as primary_error:
        if fallback_tokenizer_spec is None:
            raise
        fallback_model, fallback_revision = fallback_tokenizer_spec
        try:
            tokenizer = auto_tokenizer.from_pretrained(
                fallback_model,
                revision=fallback_revision,
                use_fast=False,
                local_files_only=local_files_only,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "could not load either the model tokenizer or the explicitly requested pinned fallback; "
                f"model tokenizer error={primary_error!r}; fallback error={fallback_error!r}"
            ) from fallback_error
        return tokenizer, {
            "tokenizer_model": fallback_model,
            "tokenizer_revision": fallback_revision,
            "tokenizer_source": "explicit_pinned_fallback",
            "model_tokenizer_error": repr(primary_error),
        }


def _load_model_with_legacy_bert_fallback(
    auto_model: object,
    bert_config: object,
    bert_model: object,
    *,
    model_name: str,
    revision: str,
    local_files_only: bool,
) -> tuple[object, dict[str, object]]:
    """Load a model, with an explicit BERT route for legacy prajjwal configs.

    Modern ``AutoModel`` dispatch needs ``model_type``.  The pinned historical
    prajjwal configs omit it even though their weights are standard BERT
    checkpoints.  Restricting the fallback to that named family prevents an
    accidental architecture substitution for another model.
    """
    try:
        model = auto_model.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        return model, {"model_loader": "AutoModel"}
    except Exception as auto_error:
        if not model_name.startswith("prajjwal1/bert-"):
            raise
        try:
            config = bert_config.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
            model = bert_model.from_pretrained(
                model_name,
                revision=revision,
                config=config,
                local_files_only=local_files_only,
            )
        except Exception as bert_error:
            raise RuntimeError(
                "AutoModel could not load the legacy prajjwal BERT snapshot and the explicit "
                f"BertConfig/BertModel fallback also failed; auto error={auto_error!r}; "
                f"BERT fallback error={bert_error!r}"
            ) from bert_error
        return model, {
            "model_loader": "BertConfig_BertModel_legacy_fallback",
            "auto_model_error": repr(auto_error),
            "config_class": type(config).__name__,
        }


@dataclass(frozen=True)
class LinearCaptureContext:
    rows_per_module: int
    outputs_per_module: int
    model_name: str
    revision: str
    prompt_sha256: str
    attention_mask_sha256: str
    tokenizer_provenance: Mapping[str, object]
    model_provenance: Mapping[str, object]
    torch_version: str
    transformers_version: str


def _captured_linear_trace(
    *,
    name: str,
    activation: np.ndarray,
    module: object,
    attention_mask: np.ndarray,
    context: LinearCaptureContext,
) -> OperandTrace:
    if activation.shape[-1] != module.in_features:
        raise ValueError(
            f"did not capture a compatible activation for {name}: "
            f"got {list(activation.shape)}, expected final width {module.in_features}"
        )
    selected_x, valid_coordinates, selected_coordinates, selected_flat_indices, selected_indices = _select_valid_token_rows(
        activation, attention_mask, rows_per_module=context.rows_per_module,
    )
    weight = module.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    output_indices = _select_evenly(weight.shape[0], context.outputs_per_module)
    a = np.repeat(selected_x, len(output_indices), axis=0)
    b = np.tile(weight[output_indices], (len(selected_indices), 1))
    reference = np.einsum("ij,ij->i", a, b, dtype=np.float64).astype(np.float32)
    layer_match = re.search(r"(?:encoder|transformer)\.layer\.(\d+)", name)
    return OperandTrace(
        trace_id=_safe_id(f"{context.model_name}-{context.revision[:8]}-{name}"),
        source="pretrained_transformer",
        a=a,
        b=b,
        reference=reference,
        metadata={
            "model": context.model_name,
            "revision": context.revision,
            "module": name,
            "layer": int(layer_match.group(1)) if layer_match else None,
            "operation": "torch.nn.Linear: X @ W.T",
            "activation_shape": list(activation.shape),
            "weight_shape": list(weight.shape),
            "contracted_width": int(weight.shape[1]),
            "row_indices": selected_flat_indices.tolist(),
            "row_index_space": "flattened_activation_rows",
            "selected_indices_within_valid_token_rows": selected_indices.tolist(),
            "valid_tokens_only": True,
            "valid_token_count": int(len(valid_coordinates)),
            "valid_token_coordinates": valid_coordinates.tolist(),
            "selected_token_coordinates": selected_coordinates.tolist(),
            "selected_flattened_row_indices": selected_flat_indices.tolist(),
            "dot_token_coordinates": np.repeat(selected_coordinates, len(output_indices), axis=0).tolist(),
            "dot_output_indices": np.tile(output_indices, len(selected_coordinates)).tolist(),
            "output_indices": output_indices.tolist(),
            "rows_per_module": int(len(selected_indices)),
            "outputs_per_module": int(len(output_indices)),
            "torch": context.torch_version,
            "transformers": context.transformers_version,
            "prompt_sha256": context.prompt_sha256,
            "attention_mask_sha256": context.attention_mask_sha256,
            **context.tokenizer_provenance,
            **context.model_provenance,
        },
    )


def capture_pretrained_linear_traces(
    *,
    model_specs: Iterable[tuple[str, str]] = DEFAULT_MODEL_SPECS,
    prompts: Iterable[str] = DEFAULT_PROMPTS,
    rows_per_module: int = 8,
    outputs_per_module: int = 8,
    local_files_only: bool = False,
    fallback_tokenizer_spec: tuple[str, str] | None = None,
) -> tuple[list[OperandTrace], list[dict[str, object]]]:
    """Capture small, exact-revision transformer linear-GEMM operand traces.

    A capture failure is represented in the status list, never replaced with a
    synthetic trace.  That distinction is critical when a runner is offline.
    ``fallback_tokenizer_spec`` is opt-in and must contain an exact
    ``(model_name, revision)`` pair; its use is recorded in both status and
    trace metadata.
    """
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
    except ImportError as error:  # pragma: no cover - dependency-dependent path
        return [], [{"status": "unavailable", "reason": repr(error), "dependency": "torch + transformers"}]

    prompt_list = list(prompts)
    if rows_per_module < 1 or outputs_per_module < 1:
        raise ValueError("rows_per_module and outputs_per_module must be positive")
    traces: list[OperandTrace] = []
    statuses: list[dict[str, object]] = []
    for model_name, revision in model_specs:
        status: dict[str, object] = {
            "model": model_name,
            "revision": revision,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "local_files_only": local_files_only,
            "prompts": prompt_list,
            "prompt_sha256": _sha256_json(prompt_list),
            "fallback_tokenizer_spec": list(fallback_tokenizer_spec) if fallback_tokenizer_spec is not None else None,
        }
        try:
            tokenizer, tokenizer_provenance = _load_tokenizer_with_optional_fallback(
                AutoTokenizer,
                model_name=model_name,
                revision=revision,
                fallback_tokenizer_spec=fallback_tokenizer_spec,
                local_files_only=local_files_only,
            )
            # Preserve tokenizer provenance even if the subsequent model load
            # is unavailable (for example a config-only offline snapshot).
            status.update(tokenizer_provenance)
            model, model_provenance = _load_model_with_legacy_bert_fallback(
                AutoModel,
                BertConfig,
                BertModel,
                model_name=model_name,
                revision=revision,
                local_files_only=local_files_only,
            )
            model = model.eval()
            named_modules = dict(model.named_modules())
            selected_names = _linear_module_names(model)
            captures: dict[str, np.ndarray] = {}
            model_traces: list[OperandTrace] = []
            hooks = []
            for name in selected_names:
                def capture(module: object, args: tuple[object, ...], *, capture_name: str = name) -> None:
                    value = args[0]
                    captures[capture_name] = value.detach().cpu().numpy().astype(np.float32, copy=True)

                hooks.append(named_modules[name].register_forward_pre_hook(capture))
            inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=32)
            try:
                attention_mask_tensor = inputs.get("attention_mask")
                if attention_mask_tensor is None:
                    raise ValueError("tokenizer did not return attention_mask; refusing to capture padded token rows")
                attention_mask = attention_mask_tensor.detach().cpu().numpy().astype(bool, copy=False)
                with torch.no_grad():
                    model(**inputs)
            finally:
                for hook in hooks:
                    hook.remove()

            attention_mask_sha256 = _sha256_json(attention_mask.astype(np.uint8).tolist())
            prompt_sha256 = _sha256_json(prompt_list)
            for name in selected_names:
                x = captures.get(name)
                module = named_modules[name]
                if x is None:
                    raise ValueError(
                        f"did not capture a compatible activation for {name}: got None, "
                        f"expected final width {module.in_features}"
                    )
                model_traces.append(
                    _captured_linear_trace(
                        name=name,
                        activation=x,
                        module=module,
                        attention_mask=attention_mask,
                        context=LinearCaptureContext(
                            rows_per_module=rows_per_module,
                            outputs_per_module=outputs_per_module,
                            model_name=model_name,
                            revision=revision,
                            prompt_sha256=prompt_sha256,
                            attention_mask_sha256=attention_mask_sha256,
                            tokenizer_provenance=tokenizer_provenance,
                            model_provenance=model_provenance,
                            torch_version=torch.__version__,
                            transformers_version=transformers.__version__,
                        ),
                    )
                )
            status.update(
                {
                    "status": "captured",
                    "input_shape": list(inputs["input_ids"].shape),
                    "attention_mask_shape": list(attention_mask.shape),
                    "attention_mask_sha256": _sha256_json(attention_mask.astype(np.uint8).tolist()),
                    "input_ids_sha256": _sha256_json(inputs["input_ids"].detach().cpu().tolist()),
                    "valid_tokens_only": True,
                    "valid_token_count": int(np.count_nonzero(attention_mask)),
                    "modules": selected_names,
                    "trace_count": len(model_traces),
                    "model_type": getattr(model.config, "model_type", None),
                    **tokenizer_provenance,
                    **model_provenance,
                }
            )
            traces.extend(model_traces)
        except Exception as error:  # pragma: no cover - network/model dependent
            status.update({"status": "unavailable", "error": repr(error)})
        statuses.append(status)
    return traces, statuses


def window_trace(trace: OperandTrace, width: int, *, seed: int = 146) -> OperandTrace | None:
    """Take deterministic contiguous K-windows from true GEMM dots.

    A window is labelled as a *sub-dot* rather than a new full GEMM.  Widths
    exceeding an original contracted dimension return ``None``; callers must
    never relabel repeated/stitched operands as an actual wide transformer GEMM.
    """
    if width > trace.width:
        return None
    if width == trace.width:
        return trace
    starts = np.asarray(
        [_stable_seed(seed, trace.trace_id, width, dot) % (trace.width - width + 1) for dot in range(trace.dot_count)],
        dtype=np.int64,
    )
    offsets = np.arange(width, dtype=np.int64)[None, :]
    positions = starts[:, None] + offsets
    dots = np.arange(trace.dot_count, dtype=np.int64)[:, None]
    a = trace.a[dots, positions]
    b = trace.b[dots, positions]
    reference = np.einsum("ij,ij->i", a, b, dtype=np.float64).astype(np.float32)
    metadata = dict(trace.metadata)
    metadata.update({"subdot": True, "source_width": trace.width, "window_width": width, "window_starts": starts.tolist()})
    return OperandTrace(f"{trace.trace_id}-w{width}", trace.source, a, b, reference, metadata)


def save_operand_traces(traces: Iterable[OperandTrace], statuses: Iterable[dict[str, object]], output: Path) -> dict[str, object]:
    """Write one compact compressed shard per trace plus a deterministic manifest."""
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for trace in traces:
        filename = f"{_safe_id(trace.trace_id)}.npz"
        np.savez_compressed(output / filename, a=trace.a.astype(np.float32), b=trace.b.astype(np.float32), reference=trace.reference.astype(np.float32))
        digest = _sha256_file(output / filename)
        entries.append(
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "file": filename,
                "sha256": digest,
                "dot_count": trace.dot_count,
                "width": trace.width,
                "metadata": trace.metadata,
            }
        )
    manifest = {
        "format_version": TRACE_FORMAT_VERSION,
        "entries": entries,
        "capture_status": list(statuses),
        "note": "Shards retain selected operand windows; regenerate product descriptors for each radix/layout/group.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_operand_traces(
    path: Path,
    *,
    verify_checksums: bool = True,
) -> tuple[list[OperandTrace], dict[str, object]]:
    """Load operand shards, verifying the recorded content hash by default.

    Version-1 manifests created before shard hashes were introduced remain
    readable: an absent ``sha256`` simply has nothing to verify.  A present
    hash is always checked unless a caller explicitly disables verification
    for a forensic/recovery workflow.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != TRACE_FORMAT_VERSION:
        raise ValueError("unsupported trace manifest version")
    traces = []
    for entry in manifest.get("entries", []):
        shard_path = path / str(entry["file"])
        if verify_checksums:
            _verify_shard_checksum(shard_path, entry)
        with np.load(shard_path, allow_pickle=False) as shard:
            traces.append(
                OperandTrace(
                    str(entry["trace_id"]),
                    str(entry["source"]),
                    shard["a"].astype(np.float32),
                    shard["b"].astype(np.float32),
                    shard["reference"].astype(np.float32),
                    dict(entry.get("metadata", {})),
                )
            )
    return traces, manifest


def build_default_trace_dataset(
    output: Path,
    *,
    include_real: bool = True,
    local_files_only: bool = False,
    seed: int = 146,
    fallback_tokenizer_spec: tuple[str, str] | None = None,
) -> dict[str, object]:
    traces = synthetic_operand_traces(seed=seed)
    statuses: list[dict[str, object]] = [{"status": "captured", "source": "synthetic", "seed": seed}]
    if include_real:
        real, real_status = capture_pretrained_linear_traces(
            local_files_only=local_files_only,
            fallback_tokenizer_spec=fallback_tokenizer_spec,
        )
        traces.extend(real)
        statuses.extend(real_status)
    return save_operand_traces(traces, statuses, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture deterministic closed-log operand traces.")
    parser.add_argument("--output", type=Path, default=Path("results/closed_log/traces"))
    parser.add_argument("--no-real", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--use-pinned-bert-tokenizer-fallback",
        action="store_true",
        help="Retry a missing legacy BERT tokenizer with the documented pinned Google BERT tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=146)
    args = parser.parse_args()
    build_default_trace_dataset(
        args.output,
        include_real=not args.no_real,
        local_files_only=args.local_files_only,
        seed=args.seed,
        fallback_tokenizer_spec=(DEFAULT_BERT_FALLBACK_TOKENIZER_SPEC if args.use_pinned_bert_tokenizer_fallback else None),
    )


if __name__ == "__main__":
    main()
