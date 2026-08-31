"""Margin head numerics.

The tests that matter most here are the NaN-resistance ones. A margin head that
produces NaN on the pathological input (cos = +/-1, which a freshly initialised
head in fp16 genuinely produces) will kill a training run hours in, and the
traceback points at the loss rather than at the missing clamp.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from frs.models.heads import (  # noqa: E402
    AdaFaceHead,
    ArcFaceHead,
    CosFaceHead,
    build_head,
)

DIM, CLASSES, BATCH = 32, 10, 8
ALL_HEADS = ("arcface", "cosface", "adaface")


def make_batch(batch=BATCH, dim=DIM, classes=CLASSES):
    emb = torch.nn.functional.normalize(torch.randn(batch, dim), dim=1)
    norms = torch.rand(batch, 1) * 30 + 5
    labels = torch.randint(0, classes, (batch,))
    return emb, norms, labels


@pytest.mark.parametrize("head_type", ALL_HEADS)
def test_forward_shape_and_finiteness(head_type):
    head = build_head({"type": head_type}, DIM, CLASSES)
    emb, norms, labels = make_batch()
    logits = head(emb, norms, labels)
    assert logits.shape == (BATCH, CLASSES)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("head_type", ALL_HEADS)
def test_gradients_flow_and_are_finite(head_type):
    head = build_head({"type": head_type}, DIM, CLASSES)
    emb, norms, labels = make_batch()
    emb.requires_grad_(True)

    loss = torch.nn.functional.cross_entropy(head(emb, norms, labels), labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0, "head received no gradient"


@pytest.mark.parametrize("head_type", ALL_HEADS)
def test_adversarial_perfectly_aligned_embedding_does_not_nan(head_type):
    """cos = 1.0 exactly -- where acos' is infinite.

    This is the single most important test in the file. A head missing the
    cosine clamp passes every other test and then produces NaN in training.
    """
    head = build_head({"type": head_type}, DIM, CLASSES)
    with torch.no_grad():
        # Make sample i point exactly along prototype i, and the negation of it.
        head.weight[0] = torch.nn.functional.normalize(torch.ones(DIM), dim=0)
    aligned = torch.nn.functional.normalize(torch.ones(2, DIM), dim=1)
    aligned[1] *= -1  # cos = -1 for the anti-aligned case
    aligned.requires_grad_(True)

    norms = torch.tensor([[20.0], [20.0]])
    labels = torch.tensor([0, 0])

    logits = head(aligned, norms, labels)
    assert torch.isfinite(logits).all(), "logits went non-finite at |cos| = 1"

    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss), "loss went non-finite at |cos| = 1"
    assert torch.isfinite(aligned.grad).all(), "gradient went non-finite at |cos| = 1"


def test_arcface_acos_and_trig_identity_agree():
    """cos(theta + m) computed two ways must match.

    The default path expands the angle-addition identity to avoid acos; this
    asserts that optimisation is not silently changing the maths.
    """
    torch.manual_seed(0)
    trig = ArcFaceHead(DIM, CLASSES, m=0.5, use_acos=False)
    acos = ArcFaceHead(DIM, CLASSES, m=0.5, use_acos=True)
    acos.load_state_dict(trig.state_dict())

    emb, norms, labels = make_batch(batch=64)
    torch.testing.assert_close(
        trig(emb, norms, labels), acos(emb, norms, labels), rtol=1e-4, atol=1e-5
    )


def test_arcface_margin_reduces_target_logit():
    """The margin must make the correct class *harder*, never easier."""
    head = ArcFaceHead(DIM, CLASSES, m=0.5, scale=1.0)
    emb, norms, labels = make_batch(batch=64)

    with torch.no_grad():
        cos = head._cosine(emb)
        logits = head(emb, norms, labels)
        idx = torch.arange(len(labels))
        assert (logits[idx, labels] <= cos[idx, labels] + 1e-6).all()


def test_cosface_applies_exact_margin():
    head = CosFaceHead(DIM, CLASSES, m=0.35, scale=1.0)
    emb, norms, labels = make_batch()
    with torch.no_grad():
        cos = head._cosine(emb)
        logits = head(emb, norms, labels)
        idx = torch.arange(len(labels))
        torch.testing.assert_close(logits[idx, labels], cos[idx, labels] - 0.35)


def test_cosface_leaves_non_target_logits_untouched():
    head = CosFaceHead(DIM, CLASSES, m=0.35, scale=1.0)
    emb, norms, labels = make_batch()
    with torch.no_grad():
        cos = head._cosine(emb)
        logits = head(emb, norms, labels)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(len(labels)), labels] = False
        torch.testing.assert_close(logits[mask], cos[mask])


def test_adaface_requires_norms():
    head = AdaFaceHead(DIM, CLASSES)
    emb, _, labels = make_batch()
    with pytest.raises(ValueError, match="requires feature norms"):
        head(emb, None, labels)


def test_adaface_ema_buffers_update_and_persist():
    head = AdaFaceHead(DIM, CLASSES, t_alpha=0.5)
    # batch_std starts at 100 deliberately: it makes the margin scaler ~0 for the
    # first ~100 steps so training begins ArcFace-like.
    assert float(head.batch_std) == pytest.approx(100.0)

    before = float(head.batch_mean)
    emb, norms, labels = make_batch(batch=32)
    head(emb, norms * 0 + 40.0, labels)
    assert float(head.batch_mean) != before, "EMA did not update"

    assert "batch_mean" in head.state_dict()
    assert "batch_std" in head.state_dict(), "EMA buffers must survive checkpointing"


def test_adaface_high_norm_gets_larger_margin():
    """The core AdaFace claim: cleaner images (higher norm) get a bigger margin.

    The margin scaler standardises each norm against the *batch* distribution,
    so the batch must contain a mix of qualities for the effect to exist at all.
    A batch of constant norm has zero std and therefore carries no quality
    signal by construction -- which is correct behaviour, not a bug.
    """
    head = AdaFaceHead(DIM, CLASSES, scale=1.0, t_alpha=1.0)
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(64, DIM), dim=1)
    labels = torch.randint(0, CLASSES, (64,))
    idx = torch.arange(64)

    # Half degraded (low norm), half clean (high norm).
    norms = torch.cat([torch.full((32, 1), 5.0), torch.full((32, 1), 60.0)])

    with torch.no_grad():
        cos = head._cosine(emb)[idx, labels]
        target = head(emb, norms, labels)[idx, labels]

    penalty = cos - target  # how much margin each sample actually received
    assert penalty[32:].mean() > penalty[:32].mean(), (
        "high-norm (clean) samples should receive a larger margin than "
        "low-norm (degraded) ones"
    )


def test_adaface_uniform_norm_batch_is_neutral():
    """A constant-norm batch has zero std, so every sample gets the same margin."""
    head = AdaFaceHead(DIM, CLASSES, scale=1.0, t_alpha=1.0)
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(32, DIM), dim=1)
    labels = torch.randint(0, CLASSES, (32,))
    idx = torch.arange(32)

    with torch.no_grad():
        cos = head._cosine(emb)[idx, labels]
        penalty = cos - head(emb, torch.full((32, 1), 20.0), labels)[idx, labels]

    assert torch.isfinite(penalty).all()
    assert penalty.std() < 1e-5, "uniform norms must produce a uniform margin"


@pytest.mark.parametrize("head_type", ALL_HEADS)
def test_weight_init_matches_arcface_reference(head_type):
    head = build_head({"type": head_type}, 512, 1000)
    assert head.weight.shape == (1000, 512)
    assert head.weight.std().item() == pytest.approx(0.01, abs=0.002)


def test_build_head_ignores_partial_fc_block():
    head = build_head(
        {"type": "arcface", "m": 0.5, "partial_fc": {"enabled": True}}, DIM, CLASSES
    )
    assert isinstance(head, ArcFaceHead)
    assert head.m == 0.5


def test_arcface_theta_beyond_pi_uses_linear_extension():
    """Samples pointing away from their prototype must not get an easier target."""
    head = ArcFaceHead(DIM, CLASSES, m=0.5, scale=1.0, easy_margin=False)
    with torch.no_grad():
        head.weight[0] = torch.nn.functional.normalize(torch.ones(DIM), dim=0)
    # Anti-aligned: theta ~ pi, so theta + m > pi.
    emb = torch.nn.functional.normalize(-torch.ones(1, DIM), dim=1)
    labels = torch.tensor([0])

    with torch.no_grad():
        cos = head._cosine(emb)[0, 0]
        target = head(emb, torch.tensor([[20.0]]), labels)[0, 0]

    assert torch.isfinite(target)
    assert target < cos, "margin must still penalise, not reward, at theta > pi - m"
    assert target == pytest.approx(float(cos - math.sin(math.pi - 0.5) * 0.5), abs=1e-4)
