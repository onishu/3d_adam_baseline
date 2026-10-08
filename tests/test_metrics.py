"""Deterministic metric tests with tiny synthetic arrays (no real results)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.metrics import (
    AUPROConfig,
    aupro,
    best_f1_threshold,
    f1_at_threshold,
    image_auroc,
    image_f1_with_threshold,
    macro_average,
    metrics_to_percent,
    pixel_auroc,
)


def test_image_auroc_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert image_auroc(y_true, y_score) == pytest.approx(1.0)


def test_image_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both normal and anomalous"):
        image_auroc(np.array([1, 1]), np.array([0.5, 0.6]))


def test_pixel_auroc_basic() -> None:
    y_true = np.array([[0, 0], [1, 1]])
    y_score = np.array([[0.0, 0.1], [0.9, 0.8]])
    assert pixel_auroc(y_true, y_score) == pytest.approx(1.0)


def test_best_f1_and_provenance() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    t, f1 = best_f1_threshold(y_true, y_score)
    assert f1 == pytest.approx(1.0)
    assert f1_at_threshold(y_true, y_score, t) == pytest.approx(1.0)

    # validation-selected threshold applied to eval
    f1_eval, info = image_f1_with_threshold(
        y_true,
        y_score,
        y_true_sel=y_true,
        y_score_sel=y_score,
        selection_source="validation",
    )
    assert info.source == "validation"
    assert f1_eval == pytest.approx(1.0)

    f1_oracle, info2 = image_f1_with_threshold(y_true, y_score)
    assert info2.source == "test_optimized"
    assert f1_oracle == pytest.approx(1.0)


def test_aupro_single_region() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    score = np.zeros((4, 4), dtype=np.float64)
    score[1:3, 1:3] = 1.0
    value = aupro([mask], [score], AUPROConfig(fpr_limit=0.3, num_thresholds=20))
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(1.0, abs=1e-6)


def test_macro_average_and_percent() -> None:
    assert macro_average([0.5, 1.0, None]) == pytest.approx(0.75)
    assert metrics_to_percent(0.976) == 97.6
    assert metrics_to_percent(None) is None
