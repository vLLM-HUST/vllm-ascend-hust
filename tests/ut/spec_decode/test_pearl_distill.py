# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.spec_decode.pearl.distill import (
    PearlDistillationConfig,
    load_pearl_distillation_checkpoint,
    collate_pearl_distillation_records,
    load_pearl_distillation_records,
    pearl_distillation_loss,
    save_pearl_distillation_checkpoint,
    train_pearl_distillation_step,
)


class _TinyStudent(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 5):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **_kwargs):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=self.proj.in_features).float()
        return {"logits": self.proj(hidden)}


def test_distillation_loss_is_finite_and_teacher_is_detached():
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    loss = pearl_distillation_loss(
        student,
        teacher,
        acceptance_mask=torch.tensor([[1, 0, 1], [0, 1, 1]]),
        config=PearlDistillationConfig(temperature=2.0, distill_weight=0.7),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert teacher.grad is None


def test_acceptance_mask_changes_weighted_objective():
    student = torch.zeros(1, 2, 3)
    teacher = torch.zeros(1, 2, 3)
    labels = torch.tensor([[0, 1]])
    unweighted = pearl_distillation_loss(
        student,
        teacher,
        labels=labels,
        config=PearlDistillationConfig(distill_weight=0, acceptance_weight=0),
    )
    weighted = pearl_distillation_loss(
        student,
        teacher,
        labels=labels,
        acceptance_mask=torch.tensor([[1, 0]]),
        config=PearlDistillationConfig(distill_weight=0, acceptance_weight=5),
    )
    assert torch.allclose(unweighted, weighted)


def test_training_step_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = _TinyStudent()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    input_ids = torch.tensor([[0, 1, 2], [2, 3, 4]])
    with torch.no_grad():
        teacher = torch.full((2, 3, 7), -2.0)
        teacher[..., 1] = 3.0
    before = train_pearl_distillation_step(model, optimizer, input_ids, teacher)
    after = train_pearl_distillation_step(model, optimizer, input_ids, teacher)
    assert after.loss < before.loss
    checkpoint = tmp_path / "student.pt"
    save_pearl_distillation_checkpoint(checkpoint, model, optimizer, step=2)
    restored = _TinyStudent()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.25)
    payload = load_pearl_distillation_checkpoint(checkpoint, restored, restored_optimizer)
    assert payload["step"] == 2
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_jsonl_trace_loader_and_collator_pad_rows(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"input_ids": [1, 2], "teacher_logits": [[1, 0, 0], [0, 1, 0]], "labels": [0, 1]}\n'
        '{"input_ids": [2], "teacher_logits": [[0, 0, 1]], "acceptance_mask": [1]}\n',
        encoding="utf-8",
    )
    records = load_pearl_distillation_records(trace)
    batch = collate_pearl_distillation_records(records)
    assert batch["input_ids"].shape == (2, 2)
    assert batch["teacher_logits"].shape == (2, 2, 3)
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert "labels" in batch and "acceptance_mask" in batch
