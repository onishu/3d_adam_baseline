# TransFusion Reproduction Notes (Phase 4 / 4.5)

## External sources

| Role | URL | Commit | License | Retrieved |
|------|-----|--------|---------|-----------|
| 3D-ADAM author bench | https://github.com/PaulMcHard/3D-ADAMBench | `58c7e108a4b1de5789486fd2917245a1907234bf` | MIT | 2026-10-08 |
| Original ECCV TransFusion | https://github.com/maticfuc/ECCV_TransFusion | `a419dd93f26fa3c140fadf6d35fdf1a79074a515` | MIT | 2026-10-08 |

Local path: `third_party/3D-ADAMBench/TransFusion_3D-ADAM` (vendored; previously NOGIT — commit pinned in docs/config).

## Diff vs original ECCV_TransFusion

Author 3D-ADAMBench fork adds/changes (high level):

- ADAM3D dataset layout (`dataset/Dataset.py` adam3d branches)
- Windows `num_workers=0`
- Category list / Experiment wiring for 3D-ADAM
- RGB+D channel handling patches in `SyntAnomGenerator` / `TransFusion`

Original `geo_utils.get_plane_mask` corner-plane logic is unchanged from ECCV and **fails on ADAM3D anomalib-subset XYZ**.

## Diff vs `src.run_transfusion`

Legacy entrypoint shells out to `Experiment.py` without:

- ADAM3D-safe plane mask
- finite fail-fast
- protocol / manifest SHA in results
- AMP control (upstream hardcodes `autocast(True)`)
- DTD completeness gate

New entrypoint: `src.benchmarks.table5.transfusion` via adapter `src.data.adapters.transfusion`.

## Protocol

Default: **`anomalib_subset`** + `configs/splits/anomalib_manifest.csv`.

- `paper_protocol_equivalent=false`
- Do **not** treat `inferred_manifest` as paper protocol
- val == test under SAME_AS_TEST

## Input geometry (320 x 256)

Author CLI: `--img-width 320 --img-height 256`.

Upstream Dataset:

- `torchvision.transforms.Resize((width, height))` → torchvision treats as **(h, w)** ⇒ tensor spatial **(320, 256)**
- `cv2.resize(..., (width, height))` → OpenCV **(W, H)** ⇒ array **(256, 320)** then depth channel is `transpose(2,1,0)` to **(1, 320, 256)**

RGB and depth channels therefore align at `(C, 320, 256)` despite the naming confusion. Tests lock this expected shape.

## NaN root cause

### Confirmed

1. **Foreground / plane mask collapse (primary data bug)**  
   - ADAM3D XYZ corners have `Z==0` (invalid).  
   - Upstream `get_plane_mask` divides by corner nonzero counts → divide-by-zero / NaN plane → **all-zero `plane_mask`**.  
   - `SyntAnomGenerator` multiplies Perlin anomaly masks by `plane_mask` → **empty synthetic anomalies**.  
   - Symptoms: `RuntimeWarning: invalid value encountered in divide` from `geo_utils.py`; `plane_mask` sum == 0.

2. **Incomplete DTD**  
   - Observed broken tree with only `datasets/dtd/images/smoke/000.jpg` (1 image).  
   - Full DTD is required for RGB texture anomalies. Adapter **fail-fast** if `<100` jpg files.

### Contributing / related

3. Upstream `loss()` hardcodes `torch.cuda.amp.autocast(True)` (AMP). Adapter disables by default (`--no-amp`).
4. Long legacy run displayed `loss=nan` for many epochs while plane masks were zero and DTD incomplete. Fresh short runs with zero plane remained finite for tens of steps; the learning signal was still **invalid**. We treat empty plane mask as a **hard error** after the patch (refuse train), not as a soft skip.

### Fix (legitimate, not NaN-hiding)

- Replace corner plane fit with **valid-point plane fit** (`get_plane_mask_adam3d`), fallback to valid-depth mask if degenerate.
- Require complete DTD.
- Fail-fast `torch.isfinite` checks around data / loss / grads / params.
- Default AMP off.

## Smoke D seed-consistency audit (Phase 4.5 / 4.5b)

### First failure (Phase 4)

- Expectation (initial): `abs(loss0 - loss1) < 1e-5` (near exact equality)
- Observed: losses ≈ `5.26087` vs `5.25789` (abs diff ≈ `3e-3`); sample stems already matched
- Failure mode: **GPU first-loss**, not sample order / dataset listing

### 10-run re-audit (Phase 4.5b)

Artifact: `results/quality/smoke_d_tolerance/smoke_d_10runs_summary.json`

| Metric | Value |
|--------|-------|
| runs | 10 (each = pair A/B) |
| max abs diff | ≈ 2.91e-2 |
| mean abs diff | ≈ 1.49e-2 |
| max rel diff | ≈ 5.52e-3 |
| sample order identical | always |
| plane/synthetic mask identical | always |
| model initial weight fingerprint | always identical |
| candidate `atol=rtol=1e-3` | **fails** |
| chosen `atol=rtol=1e-2` | **passes all 10** |

Previous Phase 4.5 constants `5e-2 / 1e-2` were wider than needed on abs; tightened to **`1e-2 / 1e-2`**.

### What is compared

| Check | Criterion | Kind |
|-------|-----------|------|
| Sample path / stem order | Exact equality | exact |
| Tensor spatial shape | Exact equality (elsewhere) | exact |
| Plane / first-batch mask | Exact (allclose) in audit | exact |
| Initial weight fingerprint | Exact | exact |
| GPU first loss | `abs_diff <= 1e-2` **or** `rel_diff <= 1e-2` | absolute **or** relative |

Constants: `SMOKE_D_LOSS_ATOL=1e-2`, `SMOKE_D_LOSS_RTOL=1e-2` in `src/benchmarks/table5/transfusion.py`.

### Cause classification

- **Not** a deleted assertion; sample-order exactness retained; NaN still fails
- Gap remains after matching seed, sample order, init weights, and plane mask → attributed to **CUDA nondeterminism inside the first training forward** (attention / reductions), not DataLoader shuffle or omitted seed checks
- Dataset file order and CPU preprocessing remain exact requirements

### Policy

- Do **not** treat Smoke D pass as bit-exact reproducibility of training
- Resume tests separately compare checkpointed optimizer/scheduler/LR and bounded parameter drift

## Foreground / plane fit (Phase 4.5b)

Root cause of 100% fallback on anomalib subset:

- XYZ TIFFs store **identical values in X, Y, and Z** (not a true organized point cloud).
- Three sampled points are collinear (`matrix_rank=1`); `get_plane_from_points` returns a zero normal (`degenerate_plane_coefficients`).
- Therefore `adam3d_valid_point_fit` / `valid_point_plane_fit_with_depth_fallback` always falls back to `valid_depth_mask`.

Pilot A/B primary condition: **`valid_depth_mask`** (explicit; equivalent to prior fallback behavior).

Diagnostics: `results/quality/transfusion_plane_fit_diagnostics_1m1.csv`.

## Checkpoint selection

Author `Experiment.py` overwrites a single model at each eval epoch (last-eval policy). Default `--checkpoint-selection-metric last` with `source: author_code`. Optional metric-based best is available but, on anomalib_subset (`val==test`), is **not** an independent test result.

## Empty GT masks (28)

Audit: `results/quality/empty_masks.csv` + `empty_masks_summary.json`.

**Metric handling**

- Image AUROC: image labels from `is_anomaly` (empty GT still anomalous)
- Pixel AUROC / AUPRO: `--empty-gt-pixel-policy` =
  - `include_empty_anomalous_masks` (default; primary until paper policy confirmed)
  - `exclude_empty_anomalous_masks` (secondary column)
- Result JSON records `pixel_metric_policy`, counts, and both policies when computed

## AUPRO dual reporting

- `author_aupro`: 3D-ADAMBench `EvaluationUtils.compute_pro` (FPR limit 0.3, score-sorted curve, 8-connectivity)
- `common_aupro`: `src.common.metrics.aupro` (threshold grid, 4-connectivity default)
- Primary `AUPRO` field mirrors **author** for Table V; never silently average/unify

## Foreground strategies

Configurable `--foreground-strategy`:

- `upstream_corner_fit` (author; often all-zero on ADAM3D)
- `adam3d_valid_point_fit` (**default**)
- `valid_depth_mask`
- `provided_foreground_mask` (falls back to valid-depth when masks missing)

Audit: `results/quality/transfusion_foreground_1m1.csv`

## Smoke commands

```powershell
uv run --system-certs pytest -m "not gpu" -q

uv run --system-certs pytest -m "gpu" -q `
  tests/test_transfusion_finite.py `
  tests/test_transfusion_shapes.py

uv run --system-certs python -m src.benchmarks.table5.transfusion `
  --root datasets/ADAM3D `
  --protocol anomalib_subset `
  --categories 1m1 `
  --smoke `
  --no-amp `
  --fail-on-nonfinite
```

## Local project changes (Phase 4)

| Path | Role |
|------|------|
| `configs/models/transfusion.yaml` | Pinned source + protocol + preprocess |
| `src/data/adapters/transfusion.py` | Adapter: plane patch, DTD gate, shapes |
| `src/common/finite_check.py` | Fail-fast `torch.isfinite` |
| `src/common/tensor_stats.py` | JSON-serializable tensor stats |
| `src/benchmarks/table5/transfusion.py` | New entrypoint (smoke A–D) |
| `src/data/audit_empty_masks.py` | Empty GT mask audit |
| `src/run_transfusion.py` | Deprecated wrapper / warning |
| `tests/test_transfusion_*.py` | Adapter / shape / finite tests |
| `docs/transfusion_reproduction.md` | This document |
| `results/quality/empty_masks.*` | Empty-mask audit outputs |

`third_party/3D-ADAMBench` is **not** edited; plane/AMP patches are runtime-only.

## DTD / foreground masks

| Asset | Status |
|-------|--------|
| DTD `datasets/dtd/images` | Ready: **5640** jpg (dtd-r1.0.1). path-manifest SHA-256 `c7161032d3e42e9ebef03649f2474ca7bdfcdd9c91caafde46660a2499155b55` |
| `fg_masks/adam3d_masks` | **Missing** (0 files). Upstream `--no-fg-masks` + plane_mask used |

## Smoke results (1m1, Phase 4)

- Smoke A–C: forward / loss / backward / optimizer.step / ckpt load / anomaly map — **finite**
- Smoke D: sample order match; first loss within 5e-2 (CUDA not bit-exact)
- 3-step losses example: ~5.37 → 5.12 → 5.00; `plane_mask_sum=6568`
- Peak GPU ~6.0 GB

## Full train (NOT started in Phase 4)

Requires user approval. Rough estimate (RTX 3060, bs=1, 1501 epochs, sequential 23 cats):

- ~0.7–1.0 s/step × ~40 train samples/cat ≈ 0.5–0.7 h/epoch-equivalent… actually ~30 s/epoch → **~10–14 h/category** → **~10–14 days** wall for 23 cats
- Disk: last-epoch ckpt (~0.2–0.5 GB/cat if one ckpt) + logs/metrics → budget **~15–30 GB** for 23 cats (eval every 750 epochs keeps few ckpts)

```powershell
# After user approval: unlock --full-train in table5/transfusion.py first
# (Phase 4 currently refuses full train and defaults to --smoke).
uv run --system-certs python -m src.benchmarks.table5.transfusion `
  --root datasets/ADAM3D `
  --protocol anomalib_subset `
  --categories all `
  --full-train `
  --no-amp `
  --fail-on-nonfinite `
  --output-dir results/anomalib_protocol/transfusion
```

Do **not** use legacy `src.run_transfusion` for full train until it applies the same plane-mask / AMP / finite patches.
