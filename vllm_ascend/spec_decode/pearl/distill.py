# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL-2 teacher/student distillation primitives.

The upstream PEARL runtime is inference-focused and does not provide a
model-specific training loop.  This module keeps the training contract small
enough to reuse with Qwen/Llama students while leaving dataloading and model
construction to the caller.  All operations are regular PyTorch operators, so
the same code runs on CPU for tests and on Ascend when the model is placed on an
NPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PearlDistillationConfig:
    """Numerical policy for one PEARL-2 distillation step."""

    temperature: float = 1.0
    distill_weight: float = 0.5
    acceptance_weight: float = 1.0
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("Distillation temperature must be positive.")
        if not 0 <= self.distill_weight <= 1:
            raise ValueError("Distill weight must be in [0, 1].")
        if self.acceptance_weight < 0:
            raise ValueError("Acceptance weight must be non-negative.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("Label smoothing must be in [0, 1).")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("Gradient clipping norm must be positive when supplied.")


@dataclass(frozen=True)
class PearlDistillationMetrics:
    """Detached scalar metrics emitted by :func:`train_pearl_distillation_step`."""

    loss: float
    distill_loss: float
    hard_loss: float
    accepted_fraction: float
    tokens: int


def load_pearl_distillation_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL teacher traces used by the PEARL-2 trainer.

    Each row must contain ``input_ids`` and a ``teacher_logits`` matrix with
    shape ``[tokens, vocab]``. Optional ``labels`` and ``acceptance_mask``
    fields are kept as lists and validated during collation.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(record, dict) or "input_ids" not in record or "teacher_logits" not in record:
                raise ValueError("Every distillation record needs input_ids and teacher_logits")
            records.append(record)
    if not records:
        raise ValueError("Distillation trace is empty")
    return records


def collate_pearl_distillation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Pad trace rows into tensors accepted by :func:`train_pearl_distillation_step`."""

    rows = list(records)
    if not rows:
        raise ValueError("Cannot collate an empty distillation batch")
    lengths: list[int] = []
    vocab_size: int | None = None
    for row in rows:
        input_ids = row.get("input_ids")
        logits = row.get("teacher_logits")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError("input_ids must be a non-empty list")
        if not isinstance(logits, list) or len(logits) != len(input_ids):
            raise ValueError("teacher_logits must have one [vocab] row per input token")
        if any(not isinstance(values, list) or not values for values in logits):
            raise ValueError("teacher_logits rows must be non-empty lists")
        row_vocab = len(logits[0])
        if any(len(values) != row_vocab for values in logits):
            raise ValueError("teacher_logits rows must have a constant vocabulary width")
        if vocab_size is None:
            vocab_size = row_vocab
        elif vocab_size != row_vocab:
            raise ValueError("all distillation records must use the same vocabulary width")
        lengths.append(len(input_ids))
    assert vocab_size is not None
    max_length = max(lengths)
    resolved_device = device or "cpu"
    input_tensor = torch.zeros((len(rows), max_length), dtype=torch.long, device=resolved_device)
    teacher_tensor = torch.zeros(
        (len(rows), max_length, vocab_size), dtype=torch.float32, device=resolved_device
    )
    attention = torch.zeros((len(rows), max_length), dtype=torch.bool, device=resolved_device)
    labels = torch.zeros((len(rows), max_length), dtype=torch.long, device=resolved_device)
    acceptance = torch.zeros((len(rows), max_length), dtype=torch.bool, device=resolved_device)
    has_labels = any("labels" in row for row in rows)
    has_acceptance = any("acceptance_mask" in row for row in rows)
    for batch_index, (row, length) in enumerate(zip(rows, lengths)):
        input_tensor[batch_index, :length] = torch.tensor(row["input_ids"], dtype=torch.long, device=resolved_device)
        teacher_tensor[batch_index, :length] = torch.tensor(row["teacher_logits"], dtype=torch.float32, device=resolved_device)
        attention[batch_index, :length] = True
        if "labels" in row:
            values = row["labels"]
            if not isinstance(values, list) or len(values) != length:
                raise ValueError("labels must align with input_ids")
            labels[batch_index, :length] = torch.tensor(values, dtype=torch.long, device=resolved_device)
        if "acceptance_mask" in row:
            values = row["acceptance_mask"]
            if not isinstance(values, list) or len(values) != length:
                raise ValueError("acceptance_mask must align with input_ids")
            acceptance[batch_index, :length] = torch.tensor(values, dtype=torch.bool, device=resolved_device)
    batch = {
        "input_ids": input_tensor,
        "teacher_logits": teacher_tensor,
        "attention_mask": attention,
    }
    if has_labels:
        batch["labels"] = labels
    if has_acceptance:
        batch["acceptance_mask"] = acceptance
    return batch


def _validate_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("Student and teacher logits must have shape [batch, tokens, vocab].")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have identical shapes for distillation."
        )
    if student_logits.device != teacher_logits.device:
        raise ValueError("Student and teacher logits must be on the same device.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("Student and teacher logits must be floating-point tensors.")


def _token_weights(
    student_logits: torch.Tensor,
    acceptance_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    acceptance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = student_logits.shape[:2]
    device = student_logits.device
    valid = torch.ones(shape, dtype=student_logits.dtype, device=device)
    accepted = torch.ones(shape, dtype=student_logits.dtype, device=device)
    if attention_mask is not None:
        if tuple(attention_mask.shape) != shape:
            raise ValueError("attention_mask must have shape [batch, tokens].")
        valid = attention_mask.to(device=device, dtype=student_logits.dtype).clamp_(0, 1)
    if acceptance_mask is not None:
        if tuple(acceptance_mask.shape) != shape:
            raise ValueError("acceptance_mask must have shape [batch, tokens].")
        accepted = acceptance_mask.to(device=device, dtype=student_logits.dtype).clamp_(0, 1)
    weights = valid * (1.0 + float(acceptance_weight) * accepted)
    return weights, valid


def _compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    acceptance_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    config: PearlDistillationConfig,
) -> tuple[torch.Tensor, PearlDistillationMetrics]:
    _validate_logits(student_logits, teacher_logits)
    if labels is not None and tuple(labels.shape) != student_logits.shape[:2]:
        raise ValueError("labels must have shape [batch, tokens].")

    # Teacher probabilities are intentionally detached: the caller may pass
    # logits from a separately executed target worker without retaining its
    # graph, which is the normal PEARL-2 data-collection path.
    teacher = teacher_logits.detach()
    temperature = float(config.temperature)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher / temperature, dim=-1)
    token_kl = (
        F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        * (temperature * temperature)
    )
    teacher_labels = teacher.argmax(dim=-1)
    hard_labels = teacher_labels if labels is None else labels.to(device=student_logits.device)
    token_hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        hard_labels.reshape(-1).long(),
        reduction="none",
        label_smoothing=float(config.label_smoothing),
    ).reshape(student_logits.shape[:2])
    weights, valid = _token_weights(
        student_logits,
        acceptance_mask,
        attention_mask,
        config.acceptance_weight,
    )
    denominator = weights.sum().clamp_min(torch.finfo(student_logits.dtype).eps)
    distill_loss = (token_kl * weights).sum() / denominator
    hard_loss = (token_hard * weights).sum() / denominator
    loss = config.distill_weight * distill_loss + (1.0 - config.distill_weight) * hard_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("PEARL-2 distillation produced a non-finite loss.")
    valid_tokens = int(valid.sum().detach().item())
    if acceptance_mask is None:
        accepted_fraction = 1.0
    elif valid_tokens == 0:
        accepted_fraction = 0.0
    else:
        accepted_values = acceptance_mask.to(device=student_logits.device, dtype=student_logits.dtype)
        accepted_fraction = float((accepted_values * valid).sum().detach().item() / valid_tokens)
    metrics = PearlDistillationMetrics(
        loss=float(loss.detach().item()),
        distill_loss=float(distill_loss.detach().item()),
        hard_loss=float(hard_loss.detach().item()),
        accepted_fraction=accepted_fraction,
        tokens=valid_tokens,
    )
    return loss, metrics


def pearl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    acceptance_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    config: PearlDistillationConfig | None = None,
) -> torch.Tensor:
    """Return the acceptance-weighted PEARL-2 distillation loss.

    ``teacher_logits`` never receives gradients.  When hard ``labels`` are not
    supplied, the teacher argmax is used as the hard target, making the helper
    suitable for an online target-worker trace as well as an offline dataset.
    """

    loss, _ = _compute_distillation_loss(
        student_logits,
        teacher_logits,
        labels=labels,
        acceptance_mask=acceptance_mask,
        attention_mask=attention_mask,
        config=config or PearlDistillationConfig(),
    )
    return loss


def train_pearl_distillation_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    acceptance_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    config: PearlDistillationConfig | None = None,
) -> PearlDistillationMetrics:
    """Run one optimizer step for a PEARL-2 student model.

    The model may return a tensor, a ``ModelOutput`` with ``.logits``, or a
    mapping containing ``"logits"``.  This keeps the helper compatible with
    Transformers and the native PEARL model wrappers.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens].")
    cfg = config or PearlDistillationConfig()
    optimizer.zero_grad(set_to_none=True)
    kwargs: dict[str, Any] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    outputs = model(input_ids, **kwargs)
    if isinstance(outputs, torch.Tensor):
        student_logits = outputs
    elif hasattr(outputs, "logits"):
        student_logits = outputs.logits
    elif isinstance(outputs, dict) and "logits" in outputs:
        student_logits = outputs["logits"]
    else:
        raise TypeError("Student model output must be a tensor or expose a logits field.")
    loss, metrics = _compute_distillation_loss(
        student_logits,
        teacher_logits,
        labels=labels,
        acceptance_mask=acceptance_mask,
        attention_mask=attention_mask,
        config=cfg,
    )
    loss.backward()
    if cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    optimizer.step()
    return metrics


def save_pearl_distillation_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    config: PearlDistillationConfig | None = None,
) -> None:
    """Save model/optimizer state in a portable PEARL-2 checkpoint."""

    if step < 0:
        raise ValueError("Checkpoint step must be non-negative.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "vllm-ascend-specslo/pearl-distill-v1",
        "step": int(step),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config
    torch.save(payload, destination)


def load_pearl_distillation_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model and optimizer state."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "vllm-ascend-specslo/pearl-distill-v1":
        raise ValueError("Unsupported PEARL-2 distillation checkpoint format.")
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("Checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(payload["optimizer"])
    return payload


__all__ = [
    "PearlDistillationConfig",
    "PearlDistillationMetrics",
    "collate_pearl_distillation_records",
    "load_pearl_distillation_records",
    "load_pearl_distillation_checkpoint",
    "pearl_distillation_loss",
    "save_pearl_distillation_checkpoint",
    "train_pearl_distillation_step",
]
