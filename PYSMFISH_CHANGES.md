# Changes in this fork (pysmFISH)

This repository is a fork of [MouseLand/cellpose](https://github.com/MouseLand/cellpose)
used by pysmFISH, an image-analysis pipeline for imaging-based spatial
transcriptomics (to be published). It adds opt-in options to `CellposeModel`
and `CellposeModel.eval` that make 3D segmentation of large stacks faster.
With the new options at their defaults, the fork behaves as the Cellpose
release it is based on. It is not an official Cellpose release; for Cellpose
itself, its documentation and support, see the upstream repository.

The changes were developed with the help of an AI coding assistant (Claude).

## Versions

One branch per Cellpose release, our changes as commits on top of it; every
version used by pysmFISH has a tag that is never moved. The package version
is read from the tag (setuptools_scm), so a tag is the release version with a
local label, `<release>+pysmfish.<n>`; `pip show cellpose` then shows e.g.
`4.2.1.1+pysmfish.1`.

| Tag | Based on | Branch |
|---|---|---|
| `v4.2.1.1+pysmfish.1` | Cellpose `v4.2.1.1` | `pysmfish/v4.2.1.1` |

Install a tag with pip:

```bash
pip install "cellpose @ git+https://github.com/larsborm/cellpose.git@v4.2.1.1+pysmfish.1"
```

or in the `pip:` section of a conda environment file:

```yaml
  - pip:
      - cellpose @ git+https://github.com/larsborm/cellpose.git@v4.2.1.1+pysmfish.1
```

## Overview

General = useful to any Cellpose user; the column says where a change is
mainly of use. None of the code in this fork is specific to pysmFISH; the
pipeline-specific code stays in pysmFISH (see "What stays in pysmFISH").

| ID | Change | Option | Useful for | Default | Output when on |
|---|---|---|---|---|---|
| F1 | Thin orthogonal slices packed several per tile | `pack_thin`, `pack_gap_min` | General: 3D stacks with few planes (thin or strongly anisotropic) | off | differs from unpacked (neighbouring slices share a tile) |
| F2 | Empty tiles skip the network | `skip_empty_tiles` | General: large images with empty areas | off | empty tiles get zero flows and cell probability -10 |
| F3 | Compiled network | `CellposeModel(compile=...)` | General (CUDA) | off | about 2x faster network; not equal to the bit to the uncompiled network |
| F4 | Command-line flags, docs and tests of F1-F3 | `--pack_thin`, `--pack_gap_min`, `--skip_empty_tiles`, `--compile` | General | — | — |
| F5 | 3D evaluation on the GPU | `on_device` | General: 3D on CUDA | off | equal to the bit to the default path |
| F6 | No flows returned | `return_flows` | General: callers that need only the masks | flows returned | masks unchanged, flows `None` |
| F7 | Masks before hole filling | `fill_holes` | General, mainly for callers that run the hole filling themselves (pysmFISH runs it in parallel); the same idea as upstream PR #884 | holes filled | masks before `fill_holes_and_remove_small_masks` |

Commits on `pysmfish/v4.2.1.1`: F1-F4 (with the section in `docs/do3d.rst`
and `tests/test_pack.py`); F5-F7 with `tests/test_device.py`; this file and
the note in `README.md`.

## The changes

### F1 Packing thin slices (`pack_thin`, `pack_gap_min`)

After the anisotropy resize, a 3D stack of `Lz` planes is run through the
network three times: the xy planes, the zy slices and the zx slices. The zy and
zx slices are only `Lz` rows deep, and each is centred in its own 256-row tile,
so most of their tiles are zeros. With `pack_thin=True` (eval, `core.run_3D`,
`core.run_net`), `k` consecutive slices share one tile row, separated by at
least `pack_gap_min` zero rows (default 24), and the outputs are cut apart
again. `transforms.pack_layout(Ly, bsize, gap_min, border)` takes the largest
`k` with `k * Ly + (k - 1) * gap_min <= bsize - 2 * border` and widens the gaps
to fill the tile; `core._run_net_packed` builds the tall planes. Planes larger
than a tile (the xy planes of real images) are never packed. Ignored with
`augment=True` or `rsz`.

Example: 61 planes after the resize give `k = 3` (3 x 61 + 2 x 24 = 231 <= 240
usable rows) with gaps of 28 rows, so the orthogonal passes run a third of the
tiles.

The output differs from the unpacked output, because neighbouring slices share
a tile (see Measurements). The section "Faster 3D on thin stacks" in
`docs/do3d.rst` describes the option.

### F2 Skipping empty tiles (`skip_empty_tiles`)

`skip_empty_tiles=<threshold>` (eval, `core.run_3D`, `core.run_net`): tiles
whose maximum normalized value is below the threshold are not run through the
network; they get zero flows and a cell probability of -10. With skipping (or a
fixed batch, F3) the batches are filled from the tiles that do run, so
they stay full. The count is logged once per `run_net` call as
`skipped <n> of <total> tiles below <threshold>`; pysmFISH reads this line, so
its format is kept.

### F3 Compiled network (`CellposeModel(compile=...)`)

`compile=True` wraps the network in `torch.compile(dynamic=False)` in its
default mode; a string selects the mode (e.g.
`compile="max-autotune-no-cudagraphs"`). CUDA only (without CUDA a warning,
and the network runs uncompiled), inference only. The input shape must stay
fixed, so partial batches are zero-padded to `batch_size` and only the first
rows of the output are kept (`core._forward(net, x, fixed_batch)`). The first
forward pass compiles (about 30 s in the default mode, minutes with
max-autotune); a new batch size or tile size recompiles.

The default mode gave the same output in separate runs (with an empty and with
a filled compile cache). Max-autotune chooses its kernels by timing them, so
two runs can differ. The network computes in bfloat16, and no compiled mode is
equal to the bit to the uncompiled network (see Measurements).

### F5 3D evaluation on the GPU (`on_device=True`)

For `do_3D=True` on a CUDA device, eval uploads the (resized) stack once and
keeps the tiles, the network outputs, the tile blending, the flow sums of the
three orientations and the mask computation on the GPU; only the labels come
back. The masks, flows and cell probability are equal to the bit to the
default path. Without `do_3D`, with `augment=True` or without CUDA, a warning
is logged and the default path runs.

- `core.tile_starts`: the tile starts of `transforms.make_tiles`.
- `core._run_tiles_device`: the tile loop of `run_net` on the GPU: padding,
  tiles in `make_tiles` order, the test of F2 for all planes before the
  loop, batches with the padding of F3, the blending of `average_tiles`.
  There is no host synchronization inside the loop, so the CPU thread only
  queues GPU work.
- `core.run_net_device`, `core.run_3D_device`: `run_net` and `run_3D`; packing
  (F1) builds the tall planes on the GPU; the flows are one tensor
  `[4, Lz, Ly, Lx]` (z, y, x flow, cell probability).
- `dynamics.compute_masks` with tensors calls `_compute_masks_device`;
  `get_masks_torch` ends in `_label_from_seeds_device` (labels placed at the
  cell pixels, too-large masks removed, labels renumbered in order of first
  appearance, `uint16` below 65,536 masks, else `uint32`).
- `dynamics.steps_interp` is split into the setup and `_steps_interp_run`, the
  integration loop both paths use.

Styles are returned as zeros, as in the default path. With `resample=True` and
a resize inside eval (`diameter` other than 30, or `anisotropy`), the flows go
back to the CPU for the resize and the mask computation runs on the CPU as
without `on_device`; to keep everything on the GPU, resize the stack before
eval or use `resample=False`. GPU memory: about 20 bytes per voxel of the
resized stack on top of the network.

### F6 No flows returned (`return_flows=False`)

eval returns `[None, None, None]` in place of the flows: no flow RGB image
(`plot.dx_to_circ`) is made, and with `on_device` the mask computation reuses
the flows' memory.

### F7 Masks before hole filling (`fill_holes=False`)

eval returns the masks after the resize to the image size but before
`utils.fill_holes_and_remove_small_masks`, so a caller can run that step
elsewhere (another thread or process). `dynamics.resize_and_compute_masks`
and `models.CellposeModel._compute_masks` take the same argument.

## What stays in pysmFISH

pysmFISH uses F1-F7 and keeps its own code for: the resize of the stack
before eval and the nearest-neighbour resize and hole filling of the masks
after it (with `diameter=30`, `anisotropy=1`, `on_device=True`,
`return_flows=False`, `fill_holes=False`; the masks equal those of the default
path), a parallel hole filling (numba) equal to the bit to
`fill_holes_and_remove_small_masks`, separate CPU threads for these stages,
CPU pinning of the GPU thread and caches of its image stitching.

## Measurements

NVIDIA H100, DAPI stacks of tissue sections.

| What | Result |
|---|---|
| Packing, 29 x 1600 x 1600 stack, anisotropy 1.87, `pack_gap_min=24` (3 slices per tile, 39-row gaps) | orthogonal tiles 25,600 -> 8,600, network stage 3-4x faster; nuclei centroids against unpacked F1 0.97 (dense tissue) / 0.93 (sparse); gap 8: 0.96 / 0.90 |
| Network, one batch of 64 tiles | uncompiled 395 ms; `compile=True` 205 ms (first call 27 s); `compile="max-autotune-no-cudagraphs"` 190 ms (first call ~6 min) |
| Compiled against uncompiled, 33 x 4800 x 4800 stack | 89 % of the output values differ; nuclei matching F1 0.993 |
| `on_device`, 33 x 9600 x 9600 stack, packing, skipping and `compile=True` in both | default path: eval 661 s; `on_device`, `return_flows=False`, `fill_holes=False` with the resize and hole filling outside eval (by the caller, in other threads): eval 277-297 s, GPU busy 99 %, masks equal to the bit |
| GPU memory peak, same stack | default path 35.6 GiB, `on_device` 37.7 GiB |
| pysmFISH 3D nuclei segmentation, 12 stacks of up to 100 Mpx x 40 planes (with its own changes above) | 81.5 -> 35.6 min, every output equal to the bit |

## Tests

`tests/test_pack.py` and `tests/test_device.py` run on the CPU (no GPU
needed): `pack_layout`, packed against unpacked with a pointwise network, the
defaults equal to the unchanged path, `_forward` with a fixed batch,
`tile_starts` against `make_tiles`, `run_net_device` against `run_net`
(several tiles per plane, skipped tiles, fixed batch, packing), `run_3D_device`
against `run_3D`, and `compute_masks` with tensors against numpy inputs;
every comparison is `np.array_equal`.

```bash
pytest tests/test_pack.py tests/test_device.py   # the fork's tests
pytest --runslow tests                           # with the upstream suite
```

`v4.2.1.1+pysmfish.1` (NVIDIA H100, Python 3.12.11, PyTorch 2.6.0+cu124): `pytest --runslow tests` passes
every test that ran in that environment: the 45 upstream tests that Cellpose
v4.2.1.1 itself passes there, and 25 of the fork's 26 tests (including the
compiled network against the uncompiled one, within bfloat16 rounding). The
tests that use the `cpdino-vitb` model fail in both, because that model needs
the `dinov3` package, which was not installed (13 upstream tests and the
fork's `test_eval_3D_pack_shape`).

## Porting to a new Cellpose release

1. Make a branch `pysmfish/<release>` from the release tag and re-apply the
   commits of the previous `pysmfish/` branch.
2. Check the upstream changes to the functions the fork copies or extends:
   `core.run_net`, `core.run_3D`, `core._forward`, `transforms.make_tiles`,
   `transforms.average_tiles`, `transforms._taper_mask`,
   `transforms.get_pad_yx`, `transforms.resize_image(_3d)`,
   `transforms.convert_image`, `dynamics.steps_interp`,
   `dynamics.get_masks_torch`, `dynamics.compute_masks`,
   `dynamics.resize_and_compute_masks`,
   `utils.fill_holes_and_remove_small_masks`, `models.CellposeModel.eval`,
   `_run_net` and `_compute_masks`.
3. Keep the invariants of F5 (below), run both test commands above and
   compare `on_device` against the default path on a real stack
   (`np.array_equal` of masks, flows and cell probability, inside one process
   when the network is compiled).
4. Tag `<release>+pysmfish.1` (a tag that is not a valid version, such as
   `v4.2.1.1-pysmfish.1`, makes the install fail) and update the table of
   versions above.

Invariants of `on_device` (F5):

- Tile grid: `run_net` computes the number of tiles with the unclamped
  `tile_overlap`, `make_tiles` the tiles with `tile_overlap` clamped to
  [0.05, 0.5]; `_run_tiles_device` raises if the two grids differ.
- Blending: `average_tiles` adds `y * mask` (float32 times the float64 taper
  mask, so float64) into a float32 array, `acc = float32(float64(acc) +
  float64(y) * mask)` after every tile in tile order; the weight sum is float64
  and the division is float64, rounded to float32. `_run_tiles_device` repeats
  these operations as separate GPU kernels (no fused multiply-add).
- Mask flows: numpy computes `dP * (cellprob > threshold)`, then a true float32
  division by 5. On CUDA, PyTorch applies a CPU-scalar divisor as a
  multiplication by its reciprocal, so the device path divides by a
  one-element device tensor.
- `np.nonzero` and `torch.nonzero` both return C order; the start points are
  float32 of the integer coordinates.
- `fastremap.renumber` numbers labels in order of first appearance in C
  order (0 kept); the device path reproduces this from the cell pixels, which
  are in C order.
- CUDA scatter (`index_put`) has no `uint16` kernel: labels are placed as
  `int32` and converted at the end.
