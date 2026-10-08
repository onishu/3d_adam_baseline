"""Finite-check and GPU smoke tests for TransFusion adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.finite_check import NonFiniteError, assert_finite
from src.data.adapters import transfusion as tf_adapter


def test_assert_finite_raises() -> None:
    with pytest.raises(NonFiniteError):
        assert_finite("x", torch.tensor([1.0, float("nan")]), enabled=True)


def test_assert_finite_disabled() -> None:
    stats = assert_finite("x", torch.tensor([float("inf")]), enabled=False)
    assert stats["inf_count"] == 1


@pytest.mark.gpu
@pytest.mark.dataset
def test_gpu_smoke_abc_finite() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Resolve before ensure_tf_on_path() which chdirs into third_party.
    root = Path("datasets/ADAM3D").resolve()
    dtd = Path("datasets/dtd/images").resolve()
    if not (root / "1m1").is_dir():
        pytest.skip("ADAM3D missing")
    # Smoke allows a stub DTD; full train gates on ≥100 images separately.
    dtd_info = tf_adapter.validate_dtd(dtd, min_images=1)
    if not dtd_info["ok"]:
        pytest.skip(f"DTD missing: {dtd_info}")

    tf_adapter.ensure_tf_on_path()
    tf_adapter.install_plane_mask_patch()
    tf_adapter.install_amp_patch(enabled=False)

    ns = tf_adapter.build_namespace(
        data_path=root, category="1m1", dtd_path=dtd, seed=3
    )
    train = tf_adapter.load_train_dataset(ns)
    s = train[0]
    assert_finite("image", s["image"], enabled=True)
    assert float(s["plane_mask"].sum()) > 0

    net = tf_adapter.load_model(ns)
    net.train()
    x = s["image"].unsqueeze(0).cuda()
    plane = s["plane_mask"].unsqueeze(0).cuda()
    idx = torch.tensor([0], device=x.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss, *_ = net.loss(x, idx=idx, plane_mask=plane, return_loss=True)
        assert_finite("loss", loss, enabled=True)
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        opt.step()
        for p in net.parameters():
            assert torch.isfinite(p).all()

    ckpt = Path("results/debug/transfusion/_pytest_ckpt.pkl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt)
    net2 = tf_adapter.load_model(ns)
    net2.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True))
    net2.eval()
    with torch.no_grad():
        b, c, h, w = x.shape
        inp = torch.cat([x, torch.zeros(b, 1, h, w, device=x.device)], dim=1)
        t = torch.ones(b, device=x.device, dtype=torch.long)
        pn, pm, pmask = net2.model(inp, t)
        assert pn.shape[-2:] == (h, w)
        assert pmask.shape[-2:] == (h, w)
        assert torch.isfinite(pmask).all()
