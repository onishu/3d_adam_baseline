"""Deterministic anomaly metrics with explicit threshold provenance.

Definitions used by this project (see docs/metric_protocol.md):

- Image/Pixel AUROC: sklearn roc_auc_score (higher score => more anomalous)
- F1: maximize F1 on a provided threshold-selection set, then apply to eval set
- AUPRO: per-region overlap integrated up to ``fpr_limit`` (default 0.3)

AUPRO implementations differ across official repos; results using different
kernels must not be silently mixed in one column.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

AverageKind = Literal["macro", "micro"]


@dataclass(frozen=True)
class ThresholdInfo:
    """Records how a binary threshold was chosen."""

    value: float
    source: str  # validation | test_optimized | fixed | adaptive_anomalib
    metric_optimized: str | None = None
    anomalib_class: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AUPROConfig:
    fpr_limit: float = 0.3
    num_thresholds: int = 100
    interpolation: str = "linear"
    connected_components: bool = True
    definition: str = "project_default_mvtec_style"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def image_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Image-level AUROC. Requires both classes present."""
    yt = _as_1d(y_true).astype(np.int32)
    ys = _as_1d(y_score).astype(np.float64)
    if yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape} y_score={ys.shape}")
    if len(np.unique(yt)) < 2:
        raise ValueError("image_auroc requires both normal and anomalous labels")
    return float(roc_auc_score(yt, ys))


def pixel_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pixel-level AUROC over flattened maps."""
    yt = _as_1d(y_true).astype(np.int32)
    ys = _as_1d(y_score).astype(np.float64)
    if yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape} y_score={ys.shape}")
    if len(np.unique(yt)) < 2:
        raise ValueError("pixel_auroc requires both normal and anomalous pixels")
    return float(roc_auc_score(yt, ys))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximizing F1 on the provided set (oracle on that set)."""
    yt = _as_1d(y_true).astype(np.int32)
    ys = _as_1d(y_score).astype(np.float64)
    if yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape} y_score={ys.shape}")
    if len(ys) == 0:
        raise ValueError("Empty arrays for F1 threshold search")

    candidates = np.unique(ys)
    if candidates.size == 0:
        raise ValueError("No score candidates for F1 threshold search")

    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        pred = (ys >= t).astype(np.int32)
        f1 = float(f1_score(yt, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    yt = _as_1d(y_true).astype(np.int32)
    ys = _as_1d(y_score).astype(np.float64)
    pred = (ys >= threshold).astype(np.int32)
    return float(f1_score(yt, pred, zero_division=0))


def image_f1_with_threshold(
    y_true_eval: np.ndarray,
    y_score_eval: np.ndarray,
    *,
    y_true_sel: np.ndarray | None = None,
    y_score_sel: np.ndarray | None = None,
    threshold: float | None = None,
    selection_source: str = "validation",
) -> tuple[float, ThresholdInfo]:
    """Compute image F1 with explicit threshold provenance."""
    if threshold is not None:
        info = ThresholdInfo(value=float(threshold), source="fixed", metric_optimized="F1")
        return f1_at_threshold(y_true_eval, y_score_eval, threshold), info

    if y_true_sel is None or y_score_sel is None:
        # Explicit oracle on the evaluation set.
        t, _ = best_f1_threshold(y_true_eval, y_score_eval)
        info = ThresholdInfo(
            value=t,
            source="test_optimized",
            metric_optimized="F1",
            notes="Threshold selected on the evaluation set (oracle).",
        )
        return f1_at_threshold(y_true_eval, y_score_eval, t), info

    t, _ = best_f1_threshold(y_true_sel, y_score_sel)
    info = ThresholdInfo(value=t, source=selection_source, metric_optimized="F1")
    return f1_at_threshold(y_true_eval, y_score_eval, t), info


def _label_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Connected-component labeling for binary 2D masks (4-connectivity)."""
    from scipy import ndimage

    labeled, n = ndimage.label(mask.astype(bool))
    regions: list[np.ndarray] = []
    for i in range(1, n + 1):
        regions.append(labeled == i)
    return regions


def aupro(
    masks: list[np.ndarray] | np.ndarray,
    scores: list[np.ndarray] | np.ndarray,
    config: AUPROConfig | None = None,
) -> float:
    """Compute AUPRO up to ``config.fpr_limit``.

    Args:
        masks: per-image binary GT masks
        scores: per-image anomaly score maps (same shape as masks)
        config: AUPRO configuration
    """
    cfg = config or AUPROConfig()
    mask_list = list(masks) if not isinstance(masks, np.ndarray) else [m for m in masks]
    score_list = list(scores) if not isinstance(scores, np.ndarray) else [s for s in scores]
    if len(mask_list) != len(score_list):
        raise ValueError("masks and scores length mismatch")
    if not mask_list:
        raise ValueError("Empty AUPRO inputs")

    # Collect region overlaps and FPR curve over score thresholds.
    all_scores = np.concatenate([_as_1d(s) for s in score_list])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), cfg.num_thresholds)

    # Global FPR uses normal pixels only.
    normal_scores = []
    regions: list[tuple[np.ndarray, np.ndarray]] = []
    for mask, score in zip(mask_list, score_list, strict=True):
        m = np.asarray(mask).astype(bool)
        s = np.asarray(score).astype(np.float64)
        if m.shape != s.shape:
            raise ValueError(f"mask/score shape mismatch: {m.shape} vs {s.shape}")
        normal_scores.append(s[~m])
        if cfg.connected_components:
            for region in _label_regions(m):
                regions.append((region, s))
        elif m.any():
            regions.append((m, s))

    if not regions:
        raise ValueError("AUPRO requires at least one anomalous region")

    normal_scores_cat = (
        np.concatenate(normal_scores) if normal_scores else np.array([], dtype=np.float64)
    )
    n_normal = max(int(normal_scores_cat.size), 1)

    pros: list[float] = []
    fprs: list[float] = []
    for t in thresholds:
        overlaps = []
        for region, s in regions:
            pred = s >= t
            inter = np.logical_and(pred, region).sum()
            overlaps.append(float(inter) / float(region.sum()))
        pros.append(float(np.mean(overlaps)))
        if normal_scores_cat.size:
            fprs.append(float((normal_scores_cat >= t).sum()) / float(n_normal))
        else:
            fprs.append(0.0)

    # Sort by FPR ascending for integration.
    order = np.argsort(fprs)
    fpr_arr = np.asarray(fprs, dtype=np.float64)[order]
    pro_arr = np.asarray(pros, dtype=np.float64)[order]

    # Keep unique FPR points (average PRO on duplicates).
    uniq_fpr, inv = np.unique(fpr_arr, return_inverse=True)
    uniq_pro = np.zeros_like(uniq_fpr)
    counts = np.zeros_like(uniq_fpr)
    for i, idx in enumerate(inv):
        uniq_pro[idx] += pro_arr[i]
        counts[idx] += 1
    uniq_pro /= np.maximum(counts, 1)

    # Restrict to FPR <= limit and normalize by limit (MVTec-style).
    limit = float(cfg.fpr_limit)
    keep = uniq_fpr <= limit
    if not np.any(keep):
        return 0.0
    fpr_k = uniq_fpr[keep]
    pro_k = uniq_pro[keep]
    if fpr_k[0] > 0:
        fpr_k = np.insert(fpr_k, 0, 0.0)
        pro_k = np.insert(pro_k, 0, pro_k[0])
    if fpr_k[-1] < limit:
        # Linear interp to limit
        if cfg.interpolation != "linear":
            raise ValueError(f"Unsupported interpolation: {cfg.interpolation}")
        if len(fpr_k) >= 2:
            slope = (pro_k[-1] - pro_k[-2]) / (fpr_k[-1] - fpr_k[-2] + 1e-12)
            pro_limit = pro_k[-1] + slope * (limit - fpr_k[-1])
        else:
            pro_limit = pro_k[-1]
        fpr_k = np.append(fpr_k, limit)
        pro_k = np.append(pro_k, pro_limit)

    trapz = getattr(np, "trapezoid", None) or np.trapz
    area = float(trapz(pro_k, fpr_k))
    return area / limit


def author_aupro(
    anomaly_maps: list[np.ndarray] | np.ndarray,
    ground_truth_maps: list[np.ndarray] | np.ndarray,
) -> float:
    """AUPRO as implemented in 3D-ADAMBench TransFusion ``EvaluationUtils.compute_pro``.

    Differences vs :func:`aupro` (project/common):
    - Score-sorted cumulative FP/PRO curve (not fixed ``num_thresholds`` grid)
    - 8-connected components via ``structure=np.ones((3,3))``
    - Hard-coded FPR integration limit 0.3 with trapezoid + normalize
    - Empty-region images contribute zero components (may change denominator)
    """
    import sys
    from pathlib import Path

    tf_root = Path(__file__).resolve().parents[2] / "third_party" / "3D-ADAMBench" / "TransFusion_3D-ADAM"
    if str(tf_root) not in sys.path:
        sys.path.insert(0, str(tf_root))
    from utils.EvaluationUtils import compute_pro

    maps = list(anomaly_maps) if not isinstance(anomaly_maps, np.ndarray) else [m for m in anomaly_maps]
    gts = (
        list(ground_truth_maps)
        if not isinstance(ground_truth_maps, np.ndarray)
        else [g for g in ground_truth_maps]
    )
    # Author expects binary GT maps; coerce
    gts_bin = [(np.asarray(g) > 0.5).astype(np.uint8) for g in gts]
    maps_f = [np.asarray(m, dtype=np.float64) for m in maps]
    return float(compute_pro(maps_f, gts_bin))


def filter_empty_anomalous_masks(
    maps: list[np.ndarray],
    gts: list[np.ndarray],
    *,
    policy: str = "include_empty_anomalous_masks",
    image_labels: list[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Apply empty-GT policy for pixel metrics while preserving image labels elsewhere.

    ``image_labels`` should be the sample-level anomaly flags (1=anomalous). When
    omitted, falls back to ``mask.max()>0`` (cannot detect anomalous_but_not_visible).
    """
    labels = (
        [int(x) for x in image_labels]
        if image_labels is not None
        else [int(np.asarray(g).max() > 0) for g in gts]
    )
    empty_idx = [
        i for i, g in enumerate(gts) if labels[i] == 1 and int(np.asarray(g).sum()) == 0
    ]
    meta: dict[str, Any] = {
        "total_test_images": len(gts),
        "empty_gt_count": len(empty_idx),
        "pixel_metric_policy": policy,
    }
    if policy == "include_empty_anomalous_masks":
        meta["included_empty_gt_count"] = len(empty_idx)
        meta["excluded_empty_gt_count"] = 0
        return maps, gts, meta
    if policy != "exclude_empty_anomalous_masks":
        raise ValueError(f"Unknown empty GT policy: {policy}")
    keep = [i for i in range(len(gts)) if i not in set(empty_idx)]
    meta["included_empty_gt_count"] = 0
    meta["excluded_empty_gt_count"] = len(empty_idx)
    return [maps[i] for i in keep], [gts[i] for i in keep], meta


def macro_average(values: list[float | None]) -> float | None:
    """Macro average over categories; ignores None."""
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def micro_average(numerators: list[float], denominators: list[float]) -> float | None:
    """Optional micro average: sum(num)/sum(den)."""
    if len(numerators) != len(denominators):
        raise ValueError("numerators/denominators length mismatch")
    den = float(sum(denominators))
    if den <= 0:
        return None
    return float(sum(numerators) / den)


def aggregate_category_metrics(
    per_category: dict[str, dict[str, float | None]],
    metric_keys: list[str],
    kind: AverageKind = "macro",
) -> dict[str, float | None]:
    """Aggregate per-category metrics. Paper comparisons must use kind='macro'."""
    if kind != "macro":
        raise ValueError(
            "Only macro aggregation is supported here for paper comparisons. "
            "Use micro_average explicitly for optional micro stats."
        )
    out: dict[str, float | None] = {}
    for key in metric_keys:
        out[key] = macro_average([row.get(key) for row in per_category.values()])
    return out


def metrics_to_percent(value: float | None) -> float | None:
    """Convert ratio metric in [0,1] to percent points."""
    if value is None:
        return None
    return round(100.0 * float(value), 2)
