"""Tests for packing thin planes, tile skipping and fixed-batch forwards (core.run_net)."""
import numpy as np
import pytest
import torch

from cellpose import core, transforms


class PointwiseNet(torch.nn.Module):
    """Returns the first three input channels as (flow y, flow x, cellprob): any tiling,
    padding or packing must reproduce the input exactly."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, x):
        return x[:, :3], torch.zeros((x.shape[0], 256), dtype=x.dtype)


def test_pack_layout():
    assert transforms.pack_layout(54) == (3, 39)
    assert transforms.pack_layout(65) == (2, 110)
    assert transforms.pack_layout(108) == (2, 24)
    assert transforms.pack_layout(130) == (1, 0)
    assert transforms.pack_layout(240) == (1, 0)
    assert transforms.pack_layout(54, bsize=384) == (5, 24)
    assert transforms.pack_layout(64, gap_min=26) == (2, 112)
    k, gap = transforms.pack_layout(54)
    assert k * 54 + (k - 1) * gap <= 240


def test_run_net_default_off_identity():
    rng = np.random.default_rng(0)
    net = PointwiseNet()
    planes = rng.random((3, 300, 700, 3), dtype=np.float32)
    y0, _ = core.run_net(net, planes, batch_size=8, bsize=256)
    y1, _ = core.run_net(net, planes, batch_size=8, bsize=256, pack_thin=False, pack_gap_min=99,
                         skip_empty_tiles=None, fixed_batch=False)
    assert np.array_equal(y0, y1)
    assert y0.shape == (3, 300, 700, 3)
    assert np.abs(y0 - planes).max() < 1e-5


def test_run_net_packed_roundtrip():
    rng = np.random.default_rng(1)
    net = PointwiseNet()
    thin = rng.random((10, 54, 700, 3), dtype=np.float32)  # 10 planes: not a multiple of 3
    a, _ = core.run_net(net, thin, batch_size=8, bsize=256)
    b, _ = core.run_net(net, thin, batch_size=8, bsize=256, pack_thin=True)
    assert a.shape == b.shape == (10, 54, 700, 3)
    assert np.abs(a - b).max() < 1e-5
    assert np.abs(b - thin).max() < 1e-5
    c, _ = core.run_net(net, thin, batch_size=8, bsize=256, pack_thin=True, pack_gap_min=100)
    assert np.abs(c - thin).max() < 1e-5


def test_pack_disabled_with_augment(caplog):
    rng = np.random.default_rng(2)
    net = PointwiseNet()
    thin = rng.random((4, 54, 300, 3), dtype=np.float32)
    a, _ = core.run_net(net, thin, batch_size=8, bsize=256, augment=True)
    with caplog.at_level("WARNING"):
        b, _ = core.run_net(net, thin, batch_size=8, bsize=256, augment=True, pack_thin=True)
    assert np.array_equal(a, b)
    assert "packing disabled" in caplog.text


def test_run_3D_default_off_identity_and_packed():
    rng = np.random.default_rng(3)
    net = PointwiseNet()
    vol = rng.random((6, 300, 40, 3), dtype=np.float32)
    a, _ = core.run_3D(net, vol, batch_size=8, bsize=256)
    b, _ = core.run_3D(net, vol, batch_size=8, bsize=256, pack_thin=False)
    assert np.array_equal(a, b)
    c, _ = core.run_3D(net, vol, batch_size=8, bsize=256, pack_thin=True)
    assert c.shape == a.shape == (6, 300, 40, 4)
    assert np.allclose(a, c, atol=1e-5)


def test_forward_fixed_batch():
    rng = np.random.default_rng(4)
    net = PointwiseNet()
    x = rng.random((5, 3, 32, 32), dtype=np.float32)
    y0, s0 = core._forward(net, x)
    y1, s1 = core._forward(net, x, fixed_batch=8)
    assert y1.shape == y0.shape and s1.shape == s0.shape
    assert np.array_equal(y0, y1)


def test_skip_empty_tiles():
    net = PointwiseNet()
    planes = np.zeros((1, 256, 512, 3), np.float32)
    planes[0, :, 256:] = 1.0
    y, _ = core.run_net(net, planes, batch_size=8, bsize=256, skip_empty_tiles=0.5)
    assert y[0, 10, 10, 2] == -10.0  # empty left tile: no cell
    assert abs(y[0, 10, 500, 2] - 1.0) < 1e-5  # right tile was run


@pytest.mark.slow
def test_eval_3D_pack_shape(cellposemodel_fixture_2layer):
    model = cellposemodel_fixture_2layer
    vol = np.zeros((5, 64, 300, 3), np.float32)
    masks, _, _ = model.eval(vol, do_3D=True, z_axis=0, channel_axis=-1, pack_thin=True)
    assert masks.shape == (5, 64, 300)
    m0, _, _ = model.eval(vol, do_3D=True, z_axis=0, channel_axis=-1)
    m1, _, _ = model.eval(vol, do_3D=True, z_axis=0, channel_axis=-1, pack_thin=False)
    assert np.array_equal(m0, m1)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_matches_eager():
    from cellpose import models
    rng = np.random.default_rng(5)
    x = rng.random((64, 3, 256, 256), dtype=np.float32)
    eager = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    comp = models.CellposeModel(gpu=True, pretrained_model="cpsam", compile=True)
    y0, _ = core._forward(eager.net, x)
    y1, _ = core._forward(comp.net, x[:50], fixed_batch=64)
    assert y1.shape == (50,) + y0.shape[1:]
    # bfloat16 network: the compiled kernels round differently, a few values
    # differ by several bfloat16 steps
    d = np.abs(y0[:50] - y1)
    assert (d > 5e-2 + 2e-2 * np.abs(y0[:50])).mean() < 1e-4
    assert d.max() < 0.5
