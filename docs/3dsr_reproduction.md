# 3DSR reproduction notes (Phase 5 — read-only / CPU)

**Status:** investigation + adapter design only. No GPU smoke / training until after TransFusion Pilot B approval.

## Upstream

| Field | Value |
|-------|-------|
| Repository | https://github.com/VitjanZ/3DSR |
| Commit (main tip, 2026-10-08) | `6f83317837c6615f893f3107235c75edda90181a` |
| License | Apache-2.0 |
| Paper | WACV 2024 — *Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation* |

## Variants

| Script | Modality | Pretrained VQ | `in_channels` |
|--------|----------|---------------|---------------|
| `train_dsr_depth.py` | depth-only | `checkpoints/DADA_D.pckl` (~39 MB) | 1 |
| `train_dsr.py` | RGB+depth | `checkpoints/DADA_RGB_D.pckl` (~25 MB) | 4 (1 depth + 3 RGB) |

Additional checkpoint files in upstream `checkpoints/`: `DADA_dtest_.pckl`, `DADA_l1_.pckl` (likely experimental; not referenced by default train scripts).

**Table V modality (Q-MDL-05):** still **unconfirmed**. Paper Table V reports 3DSR I-AUROC 48.5 / P-AUROC 49.4 / AUPRO 18.4. Upstream ships both depth-only and RGB+D. Do not assume which column matches until author confirmation or paper appendix.

## Default train hyperparameters (README / scripts)

| Param | Value | Source |
|-------|-------|--------|
| epochs | 120 | README |
| batch size | 16 | README |
| learning rate | 0.0002 | README |
| image size | 384×384 | `train_dsr*.py` (`img_dim`) |
| scheduler | MultiStepLR at `0.80 * epochs`, gamma 0.1 | author code |
| train set length | 4000 (resampled with replacement) | `MVTecDRAEMTrainDataset.__len__` |

RTX 3060 12 GB feasibility for `bs=16` is **not assumed** — defer to later GPU smoke.

## Depth / XYZ handling (author `data_loader.py`)

1. Load XYZ TIFF via `tifffile`; take **Z = channel 2**.
2. Resize with `cv2.INTER_NEAREST` to `(W, H)` from `resize_shape=[H, W]`.
3. `get_plane_mask(xyz)` + zero-mask → foreground; `fill_plane_mask`.
4. Zero background depth; per-image min/max normalize to `[0.1, 0.9]` band; `fill_depth_map` for missing pixels.
5. RGB: BGR→RGB, `/255`, same spatial resize; train-time imgaug rotate ±15°.
6. Synthetic train anomalies: Perlin noise × plane mask (not DTD textures).

**ADAM3D anomalib-subset caveat:** XYZ TIFFs have **identical X/Y/Z channels** (not true organized point clouds). Author `get_plane_mask` (corner fit) will fail similarly to TransFusion unless patched. Adapter must use valid-depth / ADAM3D-aware plane strategy and document the difference.

## Weights download

See `scripts/download_3dsr_weights.ps1`. Prefer GitHub raw / git LFS checkout of:

- `https://raw.githubusercontent.com/VitjanZ/3DSR/main/checkpoints/DADA_D.pckl`
- `https://raw.githubusercontent.com/VitjanZ/3DSR/main/checkpoints/DADA_RGB_D.pckl`

Record SHA-256 after download (script prints hashes). Do **not** auto-download during CI unless explicitly requested.

## Local layout (planned)

```
third_party/3DSR/          # optional clone at pinned commit
configs/models/3dsr.yaml
src/data/adapters/dsr.py   # import name (file cannot be 3dsr.py)
tests/test_3dsr_adapter.py
scripts/download_3dsr_weights.ps1
```

## Open questions

1. Which modality matches 3D-ADAM Table V?
2. Does Table V use MVTec3D or ADAM3D protocol (paper says ADAM3D for Table V)?
3. Pretrained object-specific 3DSR checkpoints: README says “Will be uploaded soon” — only DADA VQ weights are public.
4. AUPRO implementation vs TransFusion / project `common` metrics.
5. Plane-mask behavior on ADAM3D X=Y=Z TIFFs.

## Gate

- No 3DSR training or GPU smoke while TransFusion Pilot B occupies the GPU.
- No claim of paper-protocol equivalence on anomalib_subset.
