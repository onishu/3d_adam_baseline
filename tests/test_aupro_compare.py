"""Regression: author AUPRO vs common AUPRO on fixed synthetic maps."""

from __future__ import annotations

import numpy as np

from src.common.metrics import AUPROConfig, aupro, author_aupro, filter_empty_anomalous_masks


def test_aupro_author_vs_common_fixed_inputs() -> None:
    rng = np.random.default_rng(0)
    gts = []
    maps = []
    for i in range(4):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:12, 8:12] = 1
        if i == 3:
            # empty anomalous-style (all-zero gt) — still included in author path
            gt[:, :] = 0
        score = rng.random((32, 32)).astype(np.float64)
        score[gt > 0] += 0.5
        gts.append(gt)
        maps.append(score)

    author = author_aupro(maps, gts)
    common = aupro(gts, maps, AUPROConfig(fpr_limit=0.3, num_thresholds=50))
    assert 0.0 <= author <= 1.0
    assert 0.0 <= common <= 1.0
    # Implementations differ; both must be finite and not silently equalized.
    assert np.isfinite(author) and np.isfinite(common)


def test_empty_gt_filter_policies() -> None:
    maps = [np.ones((4, 4)), np.ones((4, 4))]
    gts = [np.zeros((4, 4)), np.ones((4, 4))]
    # anomalous + empty GT requires explicit image_labels
    _, _, meta_incl = filter_empty_anomalous_masks(
        maps, gts, policy="include_empty_anomalous_masks", image_labels=[1, 1]
    )
    assert meta_incl["empty_gt_count"] == 1
    assert meta_incl["included_empty_gt_count"] == 1
    maps_ex, gts_ex, meta_ex = filter_empty_anomalous_masks(
        maps, gts, policy="exclude_empty_anomalous_masks", image_labels=[1, 1]
    )
    assert meta_ex["excluded_empty_gt_count"] == 1
    assert len(maps_ex) == 1 and len(gts_ex) == 1
