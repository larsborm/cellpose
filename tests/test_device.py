"""Tests for the device-resident inference (core.run_net_device / run_3D_device,
dynamics.compute_masks with tensors): equal to the bit to the numpy path."""
import numpy as np
import pytest
import torch

from cellpose import core, dynamics, transforms


class ConvNet(torch.nn.Module):
    """A fixed 3x3 convolution to 3 outputs (flow y, flow x, cellprob): the output depends on the
    tile content around every pixel, so tile borders, padding and packing all matter."""

    def __init__(self, nchan=3):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(nchan, 3, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))

    def forward(self, x):
        return self.conv(x), torch.zeros((x.shape[0], 256), dtype=x.dtype)


@pytest.mark.parametrize("Ly,Lx,overlap", [(300, 700, 0.1), (256, 256, 0.1), (1000, 523, 0.2), (240, 1300, 0.6)])
def test_tile_starts_match_make_tiles(Ly, Lx, overlap):
    img = np.zeros((1, Ly, Lx), np.float32)
    _, ysub, xsub, _, _ = transforms.make_tiles(img, bsize=256, tile_overlap=overlap)
    ys, xs, by, bx = core.tile_starts(Ly, Lx, bsize=256, tile_overlap=overlap)
    assert [y[0] for y in ysub] == [int(y) for y in ys for _ in xs]
    assert [x[0] for x in xsub] == [int(x) for _ in ys for x in xs]
    assert (by, bx) == (min(256, Ly), min(256, Lx))


def _planes(shape, seed, empty_rows=0):
    x = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if empty_rows:
        x[:, :empty_rows] = 0  # tiles below a skip threshold
    return x


@pytest.mark.parametrize("kw", [dict(), dict(skip_empty_tiles=0.5), dict(fixed_batch=True),
                                dict(skip_empty_tiles=0.5, fixed_batch=True)])
@pytest.mark.parametrize("shape", [(3, 600, 700, 3), (5, 200, 300, 3)])
def test_run_net_device_equals_run_net(shape, kw):
    net = ConvNet()
    x = _planes(shape, 1, empty_rows=300)
    y0, _ = core.run_net(net, x, batch_size=8, bsize=256, **kw)
    y1 = core.run_net_device(net, torch.from_numpy(x), batch_size=8, bsize=256, **kw)
    assert y1.shape == y0.shape
    assert np.array_equal(y1.numpy(), y0)


@pytest.mark.parametrize("kw", [dict(pack_thin=True), dict(pack_thin=True, skip_empty_tiles=0.5, fixed_batch=True)])
def test_run_net_device_packed_equals_run_net(kw):
    net = ConvNet()
    thin = _planes((10, 54, 700, 3), 2)       # 10 planes: not a multiple of k = 3
    thin[3:5] = 0
    y0, _ = core.run_net(net, thin, batch_size=8, bsize=256, **kw)
    y1 = core.run_net_device(net, torch.from_numpy(thin), batch_size=8, bsize=256, **kw)
    assert np.array_equal(y1.numpy(), y0)


def test_run_3D_device_equals_run_3D():
    net = ConvNet()
    stack = _planes((40, 290, 330, 3), 3)
    stack[:, :, :120] = 0
    kw = dict(batch_size=8, bsize=256, pack_thin=True, skip_empty_tiles=0.5, fixed_batch=True)
    y0, _ = core.run_3D(net, stack, **kw)
    y1 = core.run_3D_device(net, torch.from_numpy(stack), **kw)
    assert y1.shape == (4,) + stack.shape[:-1]
    assert np.array_equal(y1.permute(1, 2, 3, 0).numpy(), y0)


def _spheres(shape=(30, 80, 90), n=12, seed=4):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    lab = np.zeros(shape, np.uint16)
    for i in range(n):
        c = rng.uniform([5, 10, 10], np.array(shape) - [5, 10, 10])
        r = rng.uniform(4, 9)
        lab[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r] = i + 1
    return lab


@pytest.mark.parametrize("overwrite", [False, True])
def test_compute_masks_device_equals_numpy(overwrite):
    lab = _spheres()
    flows = dynamics.masks_to_flows_gpu_3d(lab, device=torch.device("cpu"))
    dP = (flows * 5).astype(np.float32)
    rng = np.random.default_rng(5)
    dP += rng.normal(0, 0.05, dP.shape).astype(np.float32)
    cellprob = np.where(lab > 0, 2.0, -2.0).astype(np.float32) + rng.normal(0, 0.5, lab.shape).astype(np.float32)
    m0 = dynamics.compute_masks(dP, cellprob, niter=100, do_3D=True)
    t = torch.from_numpy(np.concatenate([dP, cellprob[None]]))
    m1 = dynamics.compute_masks(t[:3], t[3], niter=100, do_3D=True, overwrite=overwrite)
    assert m1.dtype == m0.dtype
    assert np.array_equal(m1, m0)
    assert m0.max() > 0
