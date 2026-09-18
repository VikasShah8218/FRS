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


# ------------------------------------------------ weight decay (row collapse)


def _pfc_with_optimizer(num_classes=CLASSES, sample_rate=0.1, lr=0.5, wd=1e-2, momentum=0.0):
    """Partial-FC head + backbone + the project's own optimizer builder."""
    from frs.config import Config
    from frs.engine.optim import build_optimizer

    torch.manual_seed(0)
    backbone = TinyBackbone()
    head = PartialFC(ArcFaceHead(DIM, num_classes), sample_rate=sample_rate)
    cfg = Config._wrap({"type": "sgd", "lr": lr, "momentum": momentum, "weight_decay": wd})
    return backbone, head, build_optimizer(cfg, backbone, head)


def _step(backbone, head, optimizer, labels):
    emb, norms = backbone(torch.randn(labels.numel(), DIM))
    logits, remapped = head(emb, norms, labels)
    loss = torch.nn.functional.cross_entropy(logits, remapped)
    penalty = head.pop_decay_penalty()
    (loss + penalty if penalty is not None else loss).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_optimizer_hands_classifier_decay_to_partial_fc():
    _, head, optimizer = _pfc_with_optimizer(wd=5e-4)
    group = next(g for g in optimizer.param_groups if any(p is head.head.weight for p in g["params"]))
    assert group["weight_decay"] == 0.0, "optimizer must not decay the full classifier"
    assert head.weight_decay == 5e-4, "Partial-FC must apply the decay itself"


def test_full_classifier_keeps_optimizer_decay():
    from frs.config import Config
    from frs.engine.optim import build_optimizer

    backbone, head = TinyBackbone(), ArcFaceHead(DIM, CLASSES)
    opt = build_optimizer(Config._wrap({"type": "sgd", "lr": 0.1, "weight_decay": 5e-4}), backbone, head)
    group = next(g for g in opt.param_groups if any(p is head.weight for p in g["params"]))
    assert group["weight_decay"] == 5e-4


def test_one_step_leaves_unsampled_rows_exactly_unchanged():
    backbone, head, optimizer = _pfc_with_optimizer(momentum=0.0)
    before = head.head.weight.detach().clone()
    _step(backbone, head, optimizer, torch.randint(0, CLASSES, (BATCH,)))

    selected = set(head.last_selected.tolist())
    unsampled = [i for i in range(CLASSES) if i not in selected]
    assert unsampled, "test needs some unsampled rows"
    torch.testing.assert_close(head.head.weight.detach()[unsampled], before[unsampled], rtol=0, atol=0)
    changed = (head.head.weight.detach() - before).abs().sum(1) > 0
    assert changed[sorted(selected)].all(), "every sampled row should move"


def test_sampled_rows_receive_coupled_weight_decay():
    """Gradient of the penalty on a sampled row is exactly wd * w, as SGD's own decay."""
    _, head, _ = _pfc_with_optimizer(wd=0.03)
    emb = torch.nn.functional.normalize(torch.randn(BATCH, DIM), dim=1)
    head(emb, None, torch.randint(0, CLASSES, (BATCH,)))
    penalty = head.pop_decay_penalty()
    penalty.backward()
    rows = head.last_selected
    torch.testing.assert_close(head.head.weight.grad[rows], 0.03 * head.head.weight.detach()[rows])
    others = torch.ones(CLASSES, dtype=torch.bool)
    others[rows] = False
    assert head.head.weight.grad[others].abs().sum() == 0
    assert head.pop_decay_penalty() is None, "penalty must be consumed once"


def test_no_penalty_in_eval_mode_or_without_decay():
    _, head, _ = _pfc_with_optimizer(wd=0.0)
    emb = torch.randn(BATCH, DIM)
    head(emb, None, torch.randint(0, CLASSES, (BATCH,)))
    assert head.pop_decay_penalty() is None
    head.weight_decay = 1e-3
    head.eval()
    head(emb, None, torch.randint(0, CLASSES, (BATCH,)))
    assert head.pop_decay_penalty() is None


def test_classifier_rows_do_not_collapse_over_training():
    """Regression for the Glint360K failure: median row norm fell 30x in two epochs.

    The regime has to match the real one. On Glint360K a row that is not a
    positive gets a softmax probability of ~1/36k, so almost no gradient, and
    weight decay dominates its norm. (A toy with a few classes and tiny rows is
    the opposite regime -- gradient growth dominates -- and would not show the
    bug.) So: unit-norm rows, thousands of classes, a small sample rate, and
    labels confined to a few classes.

    The same loop runs twice: once with the old behaviour (optimizer decays the
    whole matrix every step) and once through build_optimizer. The fix must keep
    the median row near its starting norm while the old path collapses it.
    """
    from frs.config import Config
    from frs.engine.optim import build_optimizer

    dim, classes, rate, batch, steps = 128, 5000, 0.02, 8, 200
    lr, wd, momentum = 0.1, 5e-2, 0.9

    def median_norm_after(old_behaviour: bool) -> tuple[float, float]:
        torch.manual_seed(0)
        inner = ArcFaceHead(dim, classes)
        with torch.no_grad():
            inner.weight.normal_(std=dim ** -0.5)  # rows start at norm ~1
        head = PartialFC(inner, sample_rate=rate)
        unused_backbone = torch.nn.Linear(dim, dim)
        if old_behaviour:
            optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            cfg = Config._wrap({"type": "sgd", "lr": lr, "momentum": momentum, "weight_decay": wd})
            optimizer = build_optimizer(cfg, unused_backbone, head)
        start = float(inner.weight.detach().norm(dim=1).median())

        torch.manual_seed(1)
        for _ in range(steps):
            emb = torch.nn.functional.normalize(torch.randn(batch, dim), dim=1)
            labels = torch.randint(0, 10, (batch,))  # positives stay within 10 classes
            logits, remapped = head(emb, None, labels)
            loss = torch.nn.functional.cross_entropy(logits, remapped)
            penalty = head.pop_decay_penalty()
            (loss + penalty if penalty is not None else loss).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return start, float(inner.weight.detach().norm(dim=1).median())

    start, old_median = median_norm_after(old_behaviour=True)
    _, new_median = median_norm_after(old_behaviour=False)
    assert old_median < 0.1 * start, (
        f"sanity: the old behaviour should collapse rows ({start:.3f} -> {old_median:.3f})"
    )
    assert new_median > 0.6 * start, f"rows collapsed: {start:.3f} -> {new_median:.3f}"
