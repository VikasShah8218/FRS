"""Partial-FC: sampled softmax over a class subset, and its checkpoint plumbing.

The regression that matters: ``PartialFC.forward`` used to assign a sliced
tensor to the head's ``nn.Parameter`` and crash on the first step. The wrapper
now goes through ``MarginHead._weight_override`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from frs.data.adapters.base import Sample  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402
from frs.engine.checkpoint import (  # noqa: E402
    extend_head_and_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from frs.models.heads import AdaFaceHead, ArcFaceHead  # noqa: E402
from frs.models.partial_fc import (  # noqa: E402
    PartialFC,
    maybe_wrap_partial_fc,
    unwrap_partial_fc,
)

DIM, CLASSES, BATCH = 16, 200, 8


class TinyBackbone(torch.nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
        self.arch, self.input_size, self.embedding_size = "tiny", (112, 112), dim

    def forward(self, x):
        out = self.fc(x)
        norm = torch.norm(out, 2, 1, True)
        return out / norm, norm


def _batch():
    torch.manual_seed(0)
    emb = torch.randn(BATCH, DIM, requires_grad=True)
    norms = torch.rand(BATCH) * 20 + 5
    labels = torch.randint(0, CLASSES, (BATCH,))
    return emb, norms, labels


@pytest.mark.parametrize("head_cls", [ArcFaceHead, AdaFaceHead])
def test_forward_samples_subset_and_remaps_labels(head_cls):
    head = PartialFC(head_cls(DIM, CLASSES), sample_rate=0.2)
    emb, norms, labels = _batch()

    logits, remapped = head(emb, norms, labels)

    assert logits.shape[0] == BATCH
    assert logits.shape[1] <= head.num_sampled + BATCH  # positives always kept
    assert logits.shape[1] < CLASSES
    assert remapped.min() >= 0 and remapped.max() < logits.shape[1]
    # the override must be cleared after the forward
    assert head.head._weight_override is None
    assert head.head.num_classes == CLASSES


def test_gradient_reaches_only_sampled_rows():
    head = PartialFC(ArcFaceHead(DIM, CLASSES), sample_rate=0.1)
    emb, norms, labels = _batch()
    logits, remapped = head(emb, norms, labels)
    torch.nn.functional.cross_entropy(logits, remapped).backward()

    grad = head.head.weight.grad
    assert grad is not None and grad.shape == (CLASSES, DIM)
    touched = (grad.abs().sum(1) > 0).sum().item()
    assert 0 < touched <= logits.shape[1]
    # every positive class received gradient
    for label in labels.tolist():
        assert grad[label].abs().sum() > 0


def test_sample_rate_one_is_passthrough():
    inner = ArcFaceHead(DIM, CLASSES)
    head = PartialFC(inner, sample_rate=1.0)
    emb, norms, labels = _batch()
    logits, remapped = head(emb, norms, labels)
    torch.testing.assert_close(logits, inner(emb, norms, labels))
    assert torch.equal(remapped, labels)


def test_maybe_wrap_auto_threshold():
    head = ArcFaceHead(DIM, CLASSES)
    assert maybe_wrap_partial_fc(head, {"enabled": "auto"}, CLASSES) is head
    assert isinstance(maybe_wrap_partial_fc(head, {"enabled": True}, CLASSES), PartialFC)
    assert maybe_wrap_partial_fc(head, None, CLASSES) is head
    assert unwrap_partial_fc(PartialFC(head)) is head
    assert unwrap_partial_fc(head) is head


def test_checkpoint_round_trip_through_wrapper(tmp_path):
    backbone = TinyBackbone()
    head = PartialFC(AdaFaceHead(DIM, CLASSES), sample_rate=0.25)
    class_map = ClassMap.from_samples(
        [Sample(path=f"/x/{i}.jpg", identity=f"id_{i:04d}") for i in range(CLASSES)]
    )
    with torch.no_grad():
        head.head.batch_mean.fill_(12.5)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, class_map=class_map, epoch=3)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # inner-head keys, not "head.weight"; wrapper recorded as provenance only
    assert set(ckpt["head"]["state_dict"]) == {"weight", "batch_mean", "batch_std"}
    assert ckpt["head"]["type"] == "AdaFaceHead"
    assert ckpt["head"]["partial_fc"] == 0.25
    assert ckpt["head"]["num_classes"] == CLASSES

    # load into a bare head AND into a wrapped head
    bare = AdaFaceHead(DIM, CLASSES)
    load_checkpoint(path, head=bare, class_map=class_map)
    assert float(bare.batch_mean) == 12.5
    wrapped = PartialFC(AdaFaceHead(DIM, CLASSES), sample_rate=0.5)
    load_checkpoint(path, head=wrapped, class_map=class_map)
    torch.testing.assert_close(wrapped.head.weight, head.head.weight)


def test_extension_on_wrapped_head():
    head = PartialFC(ArcFaceHead(DIM, CLASSES), sample_rate=0.3)
    state = {k: v.clone() for k, v in unwrap_partial_fc(head).state_dict().items()}
    new_state, _ = extend_head_and_optimizer(
        head_state=state, optimizer_state=None, old_num_classes=CLASSES,
        new_num_classes=CLASSES + 5, head=unwrap_partial_fc(head),
    )
    assert new_state["weight"].shape == (CLASSES + 5, DIM)
    unwrap_partial_fc(head).load_state_dict(new_state)
    head.num_classes = CLASSES + 5  # the wrapper caches the count
    emb, norms, labels = _batch()
    logits, remapped = head(emb, norms, labels)
    assert logits.shape[0] == BATCH
