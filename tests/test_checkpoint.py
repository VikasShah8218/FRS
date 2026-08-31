"""Checkpoint save/load/extend round-trips.

The extension tests cover the failure people actually hit: adding identities,
loading the checkpoint, and getting a shape-mismatch RuntimeError on the first
optimiser step because the momentum buffer was never grown alongside the weight.
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
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from frs.models.heads import ArcFaceHead  # noqa: E402

DIM, CLASSES = 16, 5


class TinyBackbone(torch.nn.Module):
    """Stands in for IResNet: same (embedding, norm) return contract."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
        self.arch = "tiny"
        self.input_size = (112, 112)
        self.embedding_size = dim

    def forward(self, x):
        out = self.fc(x)
        norm = torch.norm(out, 2, 1, True)
        return out / norm, norm


def make_class_map(n=CLASSES):
    return ClassMap.from_samples(
        [Sample(path=f"/x/{i}.jpg", identity=f"id_{i:03d}") for i in range(n)],
        source="test",
    )


def build_stack(num_classes=CLASSES):
    backbone = TinyBackbone()
    head = ArcFaceHead(DIM, num_classes)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=0.1, momentum=0.9
    )
    return backbone, head, optimizer


def take_a_step(backbone, head, optimizer, num_classes=CLASSES):
    """One real training step, so optimizer state is actually populated."""
    x = torch.randn(4, DIM)
    labels = torch.randint(0, num_classes, (4,))
    emb, norms = backbone(x)
    loss = torch.nn.functional.cross_entropy(head(emb, norms, labels), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def test_roundtrip_restores_identical_weights(tmp_path):
    backbone, head, optimizer = build_stack()
    take_a_step(backbone, head, optimizer)
    class_map = make_class_map()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        backbone=backbone,
        head=head,
        optimizer=optimizer,
        class_map=class_map,
        epoch=3,
        global_step=42,
    )

    new_backbone, new_head, new_optimizer = build_stack()
    state = load_checkpoint(
        path,
        backbone=new_backbone,
        head=new_head,
        optimizer=new_optimizer,
        class_map=class_map,
    )

    assert state.epoch == 3 and state.global_step == 42
    for a, b in zip(backbone.parameters(), new_backbone.parameters()):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(head.weight, new_head.weight)
    assert not state.extended


def test_optimizer_momentum_is_restored(tmp_path):
    backbone, head, optimizer = build_stack()
    for _ in range(3):
        take_a_step(backbone, head, optimizer)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, optimizer=optimizer,
                    class_map=make_class_map())

    new_backbone, new_head, new_optimizer = build_stack()
    load_checkpoint(path, backbone=new_backbone, head=new_head, optimizer=new_optimizer)

    old_buf = optimizer.state[head.weight]["momentum_buffer"]
    new_buf = new_optimizer.state[new_head.weight]["momentum_buffer"]
    torch.testing.assert_close(old_buf, new_buf)


def test_adaface_ema_buffers_survive_roundtrip(tmp_path):
    from frs.models.heads import AdaFaceHead

    backbone = TinyBackbone()
    head = AdaFaceHead(DIM, CLASSES, t_alpha=0.5)
    emb, norms = backbone(torch.randn(8, DIM))
    head(emb, norms * 0 + 33.0, torch.randint(0, CLASSES, (8,)))

    path = tmp_path / "ada.pt"
    save_checkpoint(path, backbone=backbone, head=head, class_map=make_class_map())

    restored = AdaFaceHead(DIM, CLASSES, t_alpha=0.5)
    load_checkpoint(path, backbone=TinyBackbone(), head=restored)

    torch.testing.assert_close(head.batch_mean, restored.batch_mean)
    torch.testing.assert_close(head.batch_std, restored.batch_std)


def test_extend_preserves_old_rows_exactly():
    head_state = {"weight": torch.randn(CLASSES, DIM)}
    original = head_state["weight"].clone()

    new_state, _ = extend_head_and_optimizer(
        head_state, None, old_num_classes=CLASSES, new_num_classes=CLASSES + 3
    )

    assert new_state["weight"].shape == (CLASSES + 3, DIM)
    torch.testing.assert_close(new_state["weight"][:CLASSES], original)


def test_extend_grows_optimizer_buffers():
    """The step that is always forgotten and always crashes on the next update."""
    weight = torch.randn(CLASSES, DIM)
    optim_state = {
        "state": {0: {"momentum_buffer": torch.randn(CLASSES, DIM)}},
        "param_groups": [],
    }
    original_buf = optim_state["state"][0]["momentum_buffer"].clone()

    _, new_optim = extend_head_and_optimizer(
        {"weight": weight}, optim_state, CLASSES, CLASSES + 4
    )

    buf = new_optim["state"][0]["momentum_buffer"]
    assert buf.shape == (CLASSES + 4, DIM)
    torch.testing.assert_close(buf[:CLASSES], original_buf)
    assert torch.all(buf[CLASSES:] == 0), "new momentum rows must start at zero"


def test_extend_then_step_does_not_crash(tmp_path):
    """End-to-end: train, extend, resume, and take a real optimiser step."""
    backbone, head, optimizer = build_stack(CLASSES)
    take_a_step(backbone, head, optimizer, CLASSES)
    class_map = make_class_map(CLASSES)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, optimizer=optimizer,
                    class_map=class_map, epoch=1)

    # New data arrives with 3 additional identities.
    extended_map = ClassMap.from_dict(class_map.to_dict())
    result = extended_map.extend([f"new_{i}" for i in range(3)], source="dataset2")
    assert result.num_added == 3

    new_backbone, new_head, _ = build_stack(CLASSES)
    state = load_checkpoint(
        path,
        backbone=new_backbone,
        head=new_head,
        class_map=extended_map,
    )
    assert state.extended
    assert new_head.weight.shape == (CLASSES + 3, DIM)

    # A fresh optimizer over the resized parameter must step cleanly.
    new_optimizer = torch.optim.SGD(
        list(new_backbone.parameters()) + list(new_head.parameters()),
        lr=0.1,
        momentum=0.9,
    )
    take_a_step(new_backbone, new_head, new_optimizer, CLASSES + 3)


def test_mismatched_class_map_is_rejected(tmp_path):
    """Reordered identities would silently invalidate every learned row."""
    backbone, head, optimizer = build_stack()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, class_map=make_class_map())

    scrambled = ClassMap(["zzz_different"] + [f"id_{i:03d}" for i in range(1, CLASSES)])
    with pytest.raises(ValueError, match="class map mismatch"):
        load_checkpoint(path, backbone=TinyBackbone(), head=ArcFaceHead(DIM, CLASSES),
                        class_map=scrambled)


def test_mean_embedding_init_uses_supplied_prototypes():
    head_state = {"weight": torch.randn(CLASSES, DIM)}
    prototypes = torch.nn.functional.normalize(torch.randn(2, DIM), dim=1)

    new_state, _ = extend_head_and_optimizer(
        head_state,
        None,
        CLASSES,
        CLASSES + 2,
        init="mean_embedding",
        new_prototypes=prototypes,
    )
    torch.testing.assert_close(new_state["weight"][CLASSES:], prototypes)


def test_mean_embedding_init_requires_prototypes():
    with pytest.raises(ValueError, match="requires new_prototypes"):
        extend_head_and_optimizer(
            {"weight": torch.randn(CLASSES, DIM)}, None, CLASSES, CLASSES + 1,
            init="mean_embedding",
        )


def test_inspect_reports_contents(tmp_path):
    backbone, head, optimizer = build_stack()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, optimizer=optimizer,
                    class_map=make_class_map(), epoch=7, global_step=700)

    info = inspect_checkpoint(path)
    assert info["epoch"] == 7
    assert info["global_step"] == 700
    assert info["num_classes"] == CLASSES
    assert info["num_identities_in_map"] == CLASSES
    assert info["has_optimizer"] and info["has_rng"]


def test_atomic_write_leaves_no_temp_file(tmp_path):
    backbone, head, _ = build_stack()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, backbone=backbone, head=head, class_map=make_class_map())
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
