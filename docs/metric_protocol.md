# Metric Protocol

実装: `src/common/metrics.py`  
論文目標値: `configs/paper_metrics.yaml`  
判定閾値（プロジェクト独自）: `configs/reproduction_thresholds.yaml`

## 指標定義

| 指標 | 定義 | 備考 |
|------|------|------|
| Image AUROC | `sklearn.metrics.roc_auc_score` on image scores | 両クラス必須 |
| Pixel AUROC | 同上 on flattened pixel scores/masks | 両クラス必須 |
| Image/Pixel F1 | 二値化後の F1 | **閾値 provenance 必須** |
| AUPRO | region overlap vs FPR, integrate to `fpr_limit`, normalize by limit | 既定 FPR≤0.3 |

## 閾値 provenance（必須記録）

| source | 意味 | 保存名の例 |
|--------|------|------------|
| `validation` | validation で F1 最大化し test に適用 | `image_F1Score` |
| `test_optimized` | test 上で最適閾値（oracle） | `*_test_optimized` |
| `fixed` | 固定閾値 | 設定に値を記録 |
| `adaptive_anomalib` | anomalib adaptive threshold | クラス名も記録 |

`ValSplitMode.SAME_AS_TEST` 使用時の anomalib F1 は **test_optimized に近い**可能性がある。論文比較では混同しない。

## 集約

- 論文比較: **macro average**（カテゴリ均等）
- micro average は optional で別列
- macro と micro を同一列に混ぜない
- 23 カテゴリ未満の平均は `incomplete` / 直接比較禁止

## AUPRO

プロジェクト既定（`AUPROConfig`）:

- `fpr_limit=0.3`
- `num_thresholds=100`
- `interpolation=linear`
- `connected_components=True`（scipy ndimage, 4-connectivity）
- `definition=project_default_mvtec_style`

**未確認:** 論文・TransFusion・3DSR・GLFM 各公式実装の AUPRO が同一か。  
定義が異なる値は同一列に黙って混在させない。`metric_config.aupro.definition` に実装タグを保存する。

## 単位

- 生メトリクス: ratio ∈ [0,1]
- 論文比較表: percent points（×100）
- `configs/paper_metrics.yaml` は percent で一元管理

## 再現ステータス（独自基準）

論文公式ではない。README にも明記する。

| 条件 | status |
|------|--------|
| \|Δ\| ≤ 1.0 pt かつ 23/23 | reproduced |
| \|Δ\| ≤ 3.0 pt かつ 23/23 | approximately_reproduced |
| \|Δ\| > 3.0 pt かつ 23/23 | not_reproduced |
| 完走 < 23 | incomplete |
| protocol/定義不一致 | not_comparable |
| 依存欠損など | blocked |
