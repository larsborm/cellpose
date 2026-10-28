"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging
import numpy as np
from tqdm import trange
from . import transforms, utils

import torch

TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)


def use_gpu(gpu_number=0, use_torch=True):
    """ 
    Check if GPU is available for use.

    Args:
        gpu_number (int): The index of the GPU to be used. Default is 0.
        use_torch (bool): Whether to use PyTorch for GPU check. Default is True.

    Returns:
        bool: True if GPU is available, False otherwise.

    Raises:
        ValueError: If use_torch is False, as cellpose only runs with PyTorch now.
    """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError("cellpose only runs with PyTorch now")


def _use_gpu_torch(gpu_number=0):
    """
    Checks if CUDA or MPS is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA or MPS is available and working, False otherwise.
    """
    try:
        device = torch.device("cuda:" + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info("** TORCH CUDA version installed and working. **")
        return True
    except:
        pass
    try:
        device = torch.device('mps:' + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info('** TORCH MPS version installed and working. **')
        return True
    except:
        core_logger.info('Neither TORCH CUDA nor MPS version not installed/working.')
        return False


def assign_device(use_torch=True, gpu=False, device=0):
    """
    Assigns the device (CPU or GPU or mps) to be used for computation.

    Args:
        use_torch (bool, optional): Whether to use torch for GPU detection. Defaults to True.
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        device (int or str, optional): The device index or name to be used. Defaults to 0.

    Returns:
        torch.device, bool (True if GPU is used, False otherwise)
    """

    if isinstance(device, str):
        if device != "mps" or not(gpu and torch.backends.mps.is_available()):
            device = int(device)
    if gpu and use_gpu(gpu_number=device, use_torch=use_torch):
        try:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device}')
                core_logger.info(">>>> using GPU (CUDA)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                core_logger.info(">>>> using GPU (MPS)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
    else:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu = False
        cpu = True
    
    if cpu:
        device = torch.device("cpu")
        core_logger.info(">>>> using CPU")
        gpu = False
    return device, gpu


def _to_device(x, device, dtype=torch.float32):
    """
    Converts the input tensor or numpy array to the specified device.

    Args:
        x (torch.Tensor or numpy.ndarray): The input tensor or numpy array.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The converted tensor on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).to(device, dtype=dtype)
        return X
    else:
        return x


def _from_device(X):
    """
    Converts a PyTorch tensor from the device to a NumPy array on the CPU.

    Args:
        X (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    # The cast is so numpy conversion always works
    x = X.detach().cpu().to(torch.float32).numpy()
    return x


def _forward(net, x, fixed_batch=None):
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.
        fixed_batch (int, optional): Zero-pad the batch to this size before the
            forward pass (a compiled network needs a fixed input shape); only the
            first len(x) outputs are returned. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob) and style features.
    """
    n = x.shape[0]
    if fixed_batch is not None and n < fixed_batch:
        x = np.concatenate([x, np.zeros((fixed_batch - n,) + x.shape[1:], x.dtype)], axis=0)
    X = _to_device(x, device=net.device, dtype=net.dtype)
    net.eval()
    with torch.no_grad():
        y, style = net(X)[:2]
    del X
    y = _from_device(y)[:n]
    style = _from_device(style)[:n]
    return y, style


def run_net(net, imgi, batch_size=8, augment=False, tile_overlap=0.1, bsize=224,
            rsz=None, pack_thin=False, pack_gap_min=24, skip_empty_tiles=None,
            fixed_batch=False):
    """ 
    Run network on stack of images.
    
    (faster if augment is False)

    Args:
        net (class): cellpose network (model.net)
        imgi (np.ndarray): The input image or stack of images of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        pack_thin (bool, optional): When the planes are thinner than the tile
            (e.g. the ZY/ZX slices of a 3D stack), stack several consecutive planes
            into one tile separated by zero rows instead of padding each plane to
            bsize rows (transforms.pack_layout). Same number of tiles across,
            k times fewer tiles in total. Ignored with augment=True or rsz.
            Defaults to False (output unchanged).
        pack_gap_min (int, optional): Minimum zero rows between packed planes;
            the gap is widened to fill the tile. Defaults to 24.
        skip_empty_tiles (float, optional): Tiles whose maximum value (after
            normalization) is below this threshold are not run through the
            network: their flows are 0 and their cell probability -10. Defaults
            to None (all tiles are run).
        fixed_batch (bool, optional): Zero-pad partial batches to batch_size
            (for a compiled network). Defaults to False.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability. 
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    if pack_thin:
        if augment:
            core_logger.warning("pack_thin is not supported with augment=True, packing disabled")
        elif rsz is None:
            k, gap = transforms.pack_layout(imgi.shape[1], bsize=bsize, gap_min=pack_gap_min)
            if k > 1:
                return _run_net_packed(net, imgi, k, gap, batch_size=batch_size,
                                       tile_overlap=tile_overlap, bsize=bsize,
                                       skip_empty_tiles=skip_empty_tiles,
                                       fixed_batch=fixed_batch)
    # run network
    Lz, Ly0, Lx0, nchan = imgi.shape 
    if rsz is not None:
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0
    
    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize))
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])
    
    if augment:
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
    
    
    # run multiple slices at the same time
    ntiles = ny * nx
    # number of imgs tiled together per iteration: with skipped tiles or a fixed
    # batch, several batches per iteration keep the batches full
    per_iter = batch_size * (8 if (skip_empty_tiles is not None or fixed_batch) else 1)
    nimgs = max(1, per_iter // ntiles)
    niter = int(np.ceil(Lz / nimgs))
    ziterator = (trange(niter, file=tqdm_out, mininterval=30) 
                    if niter > 10 or Lz > 1 else range(niter))
    nout = 3
    n_skipped_total = [0]
    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
        ya[:, -1] = -10.0  # skipped tiles: no flow, no cell
        stylea = np.zeros((IMGa.shape[0], 256), "float32")
        for i, b in enumerate(inds):
            # pad image for net so Ly and Lx are divisible by 4
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2,0,1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment,
                tile_overlap=tile_overlap)
            IMGa[i * ntiles : (i+1) * ntiles] = np.reshape(IMG, 
                                            (ny * nx, nchan, ly, lx))
        
        # run network
        run_tile = np.ones(IMGa.shape[0], bool)
        if skip_empty_tiles is not None:
            run_tile = IMGa.reshape(IMGa.shape[0], -1).max(axis=1) >= skip_empty_tiles
            n_skipped_total[0] += int((~run_tile).sum())
        idx = np.flatnonzero(run_tile)
        for j in range(0, len(idx), batch_size):
            bslc = idx[j:j + batch_size]
            ya0, stylea0 = _forward(net, IMGa[bslc],
                                    fixed_batch=batch_size if fixed_batch else None)
            if j == 0:
                nout = ya0.shape[1]
            ya[bslc] = ya0
            stylea[bslc] = stylea0

        # average tiles
        for i, b in enumerate(inds):
            if i==0 and k==0:
                yf = np.zeros((Lz, nout, Ly, Lx), "float32")
                styles = np.zeros((Lz, 256), "float32")
            y = ya[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, 3, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, :imgb.shape[-2], :imgb.shape[-1]]
            # stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
            # stylei /= (stylei**2).sum()**0.5
            # styles[b] = stylei
    # slices from padding
    yf = yf[:, :, ypad1 : Ly-ypad2, xpad1 : Lx-xpad2]
    yf = yf.transpose(0,2,3,1)   
    if skip_empty_tiles is not None and n_skipped_total[0]:
        core_logger.info("skipped %d of %d tiles below %.3g" % (n_skipped_total[0], ntiles * Lz, skip_empty_tiles))
    return yf, np.array(styles)


def _run_net_packed(net, imgi, k, gap, batch_size=8, tile_overlap=0.1, bsize=224,
                    skip_empty_tiles=None, fixed_batch=False):
    """run_net on thin planes packed k per tile (see run_net pack_thin).

    Consecutive planes are stacked into tall planes of k * Ly + (k - 1) * gap
    rows, which run_net pads and tiles like any plane shorter than bsize
    (centred in the tile with the usual border), and the outputs are cut apart
    again. Planes are zero-padded to a multiple of k.

    Args:
        net (class): cellpose network.
        imgi (np.ndarray): Planes of size [Lz x Ly x Lx x nchan] with Ly < bsize.
        k (int): Planes per tile.
        gap (int): Zero rows between planes.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: as run_net ([Lz x Ly x Lx x 3], styles).
    """
    n, Ly, Lx, nchan = imgi.shape
    npad = (-n) % k
    if npad:
        imgi = np.concatenate([imgi, np.zeros((npad, Ly, Lx, nchan), imgi.dtype)], axis=0)
    m = imgi.shape[0] // k
    step = Ly + gap
    tall = np.zeros((m, k * step - gap, Lx, nchan), "float32")
    grp = imgi.reshape(m, k, Ly, Lx, nchan)
    for i in range(k):
        tall[:, i * step:i * step + Ly] = grp[:, i]
    core_logger.info("packing %d planes of %d rows: %d per tile, %d-row gaps" % (n, Ly, k, gap))
    y, styles = run_net(net, tall, batch_size=batch_size, augment=False, tile_overlap=tile_overlap,
                        bsize=bsize, rsz=None, pack_thin=False, skip_empty_tiles=skip_empty_tiles,
                        fixed_batch=fixed_batch)
    nout = y.shape[-1]
    out = np.empty((m, k, Ly, Lx, nout), "float32")
    for i in range(k):
        out[:, i] = y[:, i * step:i * step + Ly]
    return out.reshape(m * k, Ly, Lx, nout)[:n], np.zeros((n, 256), "float32")


def run_3D(net, imgs, batch_size=8, augment=False,
           tile_overlap=0.1, bsize=224, net_ortho=None,
           progress=None, pack_thin=False, pack_gap_min=24,
           skip_empty_tiles=None, fixed_batch=False):
    """ 
    Run network on image z-stack.
    
    (faster if augment is False)

    Args:
        imgs (np.ndarray): The input image stack of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        net_ortho (class, optional): cellpose network for orthogonal ZY and ZX planes. Defaults to None.
        progress (QProgressBar, optional): pyqt progress bar. Defaults to None.
        pack_thin (bool, optional): pack the thin ZY/ZX slices several per tile (see run_net). Defaults to False.
        pack_gap_min (int, optional): minimum zero rows between packed slices. Defaults to 24.
        skip_empty_tiles (float, optional): skip tiles whose maximum is below this (see run_net). Defaults to None.
        fixed_batch (bool, optional): pad partial batches for a compiled network. Defaults to False.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Z flow; y[...,1] is Y flow; y[...,2] is X flow; y[...,3] is cell probability. 
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
    for p in range(3):
        xsl = imgs.transpose(pm[p])
        # per image
        core_logger.info("running %s: %d planes of size (%d, %d)" %
                         (sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]]))
        y, style = run_net(net, xsl, batch_size=batch_size, augment=augment, 
                           bsize=bsize, tile_overlap=tile_overlap, rsz=None,
                           pack_thin=pack_thin, pack_gap_min=pack_gap_min,
                           skip_empty_tiles=skip_empty_tiles, fixed_batch=fixed_batch)
        yf[..., -1] += y[..., -1].transpose(ipm[p])
        for j in range(2):
            yf[..., cp[p][j]] += y[..., cpy[p][j]].transpose(ipm[p])
        y = None; del y
    
        if progress is not None:
            progress.setValue(25 + 15 * p)
    
    return yf, style


# ---------------------------------------------------------------------------
# Device-resident inference: the planes, the tiles, the network outputs and the
# tile blending stay on the network's device. Same tiles, batches and
# arithmetic as run_net / run_3D (augment=False, rsz=None), so the outputs are
# equal to the bit: the blending repeats average_tiles' float64 products and
# sums, rounded to float32 after every tile as numpy does.
# ---------------------------------------------------------------------------

def tile_starts(Ly, Lx, bsize=224, tile_overlap=0.1):
    """Tile starts of transforms.make_tiles (augment=False) for a padded plane of Ly x Lx.

    Returns:
        tuple: (ystart, xstart, bsizeY, bsizeX) as in make_tiles.
    """
    tile_overlap = min(0.5, max(0.05, tile_overlap))
    bsizeY, bsizeX = int(min(bsize, Ly)), int(min(bsize, Lx))
    ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
    nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
    ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
    xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)
    return ystart, xstart, bsizeY, bsizeX


def _run_tiles_device(net, Lz, plane_shape, get_planes, sink, batch_size=8, tile_overlap=0.1,
                      bsize=224, skip_empty_tiles=None, fixed_batch=False):
    """run_net's tile loop on the device for Lz planes of plane_shape (Ly0, Lx0, nchan).

    Args:
        get_planes (callable): get_planes(a, b) -> planes a..b-1 [n x Ly0 x Lx0 x nchan] on the device.
        sink (callable): sink(a, y) receives the outputs of planes a..a+n-1 [n x Ly0 x Lx0 x nout].

    Returns:
        None. Skipped tiles are logged as run_net does.
    """
    device = net.device
    Ly0, Lx0, nchan = plane_shape
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Ly0, Lx0, min_size=(bsize, bsize))
    Ly, Lx = Ly0 + ypad1 + ypad2, Lx0 + xpad1 + xpad2
    ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
    nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
    ntiles = ny * nx
    ystart, xstart, ly, lx = tile_starts(Ly, Lx, bsize=bsize, tile_overlap=tile_overlap)
    if len(ystart) * len(xstart) != ntiles:
        raise ValueError("tile grid differs from run_net's tile count")
    rows = torch.as_tensor(ystart[:, None] + np.arange(ly)[None], device=device)
    cols = torch.as_tensor(xstart[:, None] + np.arange(lx)[None], device=device)
    # taper mask and its sum over the tiles, float64 as in average_tiles
    mask = torch.from_numpy(transforms._taper_mask(ly=ly, lx=lx)).to(device)
    subs = [(slice(int(y0), int(y0) + ly), slice(int(x0), int(x0) + lx)) for y0 in ystart for x0 in xstart]
    navg = torch.zeros((Ly, Lx), dtype=torch.float64, device=device)
    for ys, xs in subs:
        navg[ys, xs] += mask

    per_iter = batch_size * (8 if (skip_empty_tiles is not None or fixed_batch) else 1)
    nimgs = max(1, per_iter // ntiles)
    niter = int(np.ceil(Lz / nimgs))
    nout = 3
    spans = [(k * nimgs, min(Lz, (k + 1) * nimgs)) for k in range(niter)]

    def tiles_of(a, b):
        # tiles in make_tiles order: plane, tile row, tile column
        planes = get_planes(a, b).permute(0, 3, 1, 2)
        imgb = torch.zeros((b - a, nchan, Ly, Lx), dtype=torch.float32, device=device)
        imgb[:, :, ypad1:ypad1 + Ly0, xpad1:xpad1 + Lx0] = planes
        del planes
        IMGa = imgb[:, :, rows[:, None, :, None], cols[None, :, None, :]]
        return IMGa.permute(0, 2, 3, 1, 4, 5).reshape((b - a) * ntiles, nchan, ly, lx)

    # the tiles to run, for every batch of planes at once: one host sync before the
    # loop instead of one per batch, so the loop queues its work without waiting
    if skip_empty_tiles is not None:
        maxima = torch.cat([tiles_of(a, b).reshape((b - a) * ntiles, -1).amax(dim=1) for a, b in spans])
        run = (maxima >= skip_empty_tiles).cpu().numpy()
        del maxima
        n_skipped = int((~run).sum())
        idx_all = [np.flatnonzero(run[a * ntiles:b * ntiles]) for a, b in spans]
    else:
        n_skipped = 0
        idx_all = [np.arange((b - a) * ntiles) for a, b in spans]
    counts = [len(i) for i in idx_all]
    idx_dev = torch.as_tensor(np.concatenate(idx_all) if idx_all else np.zeros(0, np.int64), device=device)
    offs = np.concatenate([[0], np.cumsum(counts)])
    net.eval()
    for k, (a, b) in enumerate(spans):
        nb = b - a
        IMGa = tiles_of(a, b)
        ya = torch.zeros((IMGa.shape[0], nout, ly, lx), dtype=torch.float32, device=device)
        ya[:, -1] = -10.0  # skipped tiles: no flow, no cell
        idx = idx_dev[offs[k]:offs[k + 1]]
        for j in range(0, counts[k], batch_size):
            bslc = idx[j:j + batch_size]
            X = IMGa[bslc]
            n = X.shape[0]
            if fixed_batch and n < batch_size:
                X = torch.cat([X, torch.zeros((batch_size - n,) + tuple(X.shape[1:]), dtype=X.dtype,
                                              device=device)], dim=0)
            X = X.to(net.dtype)
            with torch.no_grad():
                y = net(X)[0]
            ya[bslc] = y[:n].to(torch.float32)
            del X, y
        del IMGa
        # blend the tiles: float64 product and sum, float32 after every tile
        y = ya.reshape(nb, ntiles, nout, ly, lx)
        acc = torch.zeros((nb, nout, Ly, Lx), dtype=torch.float32, device=device)
        for t, (ys, xs) in enumerate(subs):
            prod = y[:, t].double() * mask
            acc[:, :, ys, xs] = (acc[:, :, ys, xs].double() + prod).float()
        del prod, ya, y
        out = (acc.double() / navg).float()
        del acc
        sink(a, out[:, :, ypad1:Ly - ypad2, xpad1:Lx - xpad2].permute(0, 2, 3, 1))
        del out
    if skip_empty_tiles is not None and n_skipped:
        core_logger.info("skipped %d of %d tiles below %.3g" % (n_skipped, ntiles * Lz, skip_empty_tiles))


def run_net_device(net, imgi, batch_size=8, tile_overlap=0.1, bsize=224, pack_thin=False,
                   pack_gap_min=24, skip_empty_tiles=None, fixed_batch=False, sink=None):
    """run_net (augment=False, rsz=None) with everything on the network's device.

    Args:
        imgi (torch.Tensor): Planes [Lz x Ly x Lx x nchan] (float32) on the device; a strided view
            is fine.
        sink (callable, optional): sink(a, y) receives the outputs of the consecutive planes
            a..a+n-1 as [n x Ly x Lx x nout] on the device. Without a sink the outputs are
            returned as one tensor.

    Returns:
        torch.Tensor or None: [Lz x Ly x Lx x nout] (float32) when sink is None, as run_net's yf.
    """
    Lz, Ly0, Lx0, nchan = imgi.shape
    out = None
    if sink is None:
        out = torch.zeros((Lz, Ly0, Lx0, 3), dtype=torch.float32, device=imgi.device)

        def sink(a, y):
            out[a:a + y.shape[0]] = y
    k = 1
    if pack_thin:
        k, gap = transforms.pack_layout(Ly0, bsize=bsize, gap_min=pack_gap_min)
    if k > 1:
        # thin planes packed k per tall plane with zero rows between them (see _run_net_packed)
        m = int(np.ceil(Lz / k))
        step = Ly0 + gap
        H = k * step - gap
        core_logger.info("packing %d planes of %d rows: %d per tile, %d-row gaps" % (Lz, Ly0, k, gap))

        def get_tall(a, b):
            tall = torch.zeros((b - a, H, Lx0, nchan), dtype=torch.float32, device=imgi.device)
            for i in range(k):
                p0, p1 = a * k + i, min(b * k, Lz)
                sel = torch.arange(p0, p1, k, device=imgi.device)
                n_i = len(sel)
                if n_i:
                    tall[:n_i, i * step:i * step + Ly0] = imgi[sel]
            return tall

        def sink_tall(a, y):
            planes = torch.stack([y[:, i * step:i * step + Ly0] for i in range(k)], dim=1)
            planes = planes.reshape((-1,) + tuple(planes.shape[2:]))
            n = min(planes.shape[0], Lz - a * k)
            sink(a * k, planes[:n])

        _run_tiles_device(net, m, (H, Lx0, nchan), get_tall, sink_tall, batch_size=batch_size,
                          tile_overlap=tile_overlap, bsize=bsize, skip_empty_tiles=skip_empty_tiles,
                          fixed_batch=fixed_batch)
    else:
        _run_tiles_device(net, Lz, (Ly0, Lx0, nchan), lambda a, b: imgi[a:b], sink, batch_size=batch_size,
                          tile_overlap=tile_overlap, bsize=bsize, skip_empty_tiles=skip_empty_tiles,
                          fixed_batch=fixed_batch)
    return out


def run_3D_device(net, imgs, batch_size=8, tile_overlap=0.1, bsize=224, pack_thin=False,
                  pack_gap_min=24, skip_empty_tiles=None, fixed_batch=False):
    """run_3D (augment=False) with the stack, the tiles and the flows on the network's device.

    The three orientations are added into the flows as run_3D does (YX, then ZY, then
    ZX, float32), plane batch by plane batch, so no orientation's full output is held.

    Args:
        imgs (torch.Tensor): Stack [Lz x Ly x Lx x nchan] (float32) on the device.

    Returns:
        torch.Tensor: [4 x Lz x Ly x Lx] (float32) on the device: Z, Y, X flow and cell
            probability (run_3D's yf with the channel axis first).
    """
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = tuple(imgs.shape[:-1])
    yf = torch.zeros((4, *shape), dtype=torch.float32, device=imgs.device)
    for p in range(3):
        xsl = imgs.permute(pm[p])
        core_logger.info("running %s: %d planes of size (%d, %d)" %
                         (sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]]))

        def sink(a, y, p=p):
            b = a + y.shape[0]
            if p == 0:
                tgt = lambda c: yf[c, a:b]
                src = lambda c: y[..., c]
            elif p == 1:
                tgt = lambda c: yf[c, :, a:b]
                src = lambda c: y[..., c].permute(1, 0, 2)
            else:
                tgt = lambda c: yf[c, :, :, a:b]
                src = lambda c: y[..., c].permute(1, 2, 0)
            tgt(3).add_(src(2))
            for j in range(2):
                tgt(cp[p][j]).add_(src(cpy[p][j]))

        run_net_device(net, xsl, batch_size=batch_size, tile_overlap=tile_overlap, bsize=bsize,
                       pack_thin=pack_thin, pack_gap_min=pack_gap_min,
                       skip_empty_tiles=skip_empty_tiles, fixed_batch=fixed_batch, sink=sink)
    return yf
