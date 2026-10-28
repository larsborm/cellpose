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
