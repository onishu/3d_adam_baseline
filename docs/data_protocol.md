# Data Protocol

## データセット

- ルート: `datasets/ADAM3D`
- カテゴリ数: 23（anomalib MechMind Nano subset）
- レイアウト:

```text
<category>/
  train/good/{rgb,xyz}/
  validation/good/{rgb,xyz}/          # on-disk; unused by anomalib defaults
  test/<anomaly_type>/{rgb,xyz,ground_truth}/
  calibration/                        # present for some categories
```

- RGB: `.png`、XYZ: `.tiff`、mask: `.png`
- 対応: 同一 stem

## ファイル名と part / view

| パターン | 例 | 解釈 | 確度 |
|----------|----|------|------|
| `{part}_{view}` | `000_A.png` | part_id=000, view_id=A | 仮定（high for parsing, semantic は仮定） |
| `{n}` | `000.png` | part/view 不明 | 復元不可 |

欠陥の多くは numeric-only。一部カテゴリ（例: 1m2/crack, tapa3m1/hole）は `{part}_{view}` を保持。

## Protocol 定義

### 1. official anomalib protocol (`anomalib`)

**class:** `official_anomalib_protocol`  
**実装状態:** implemented  
**結果ディレクトリ:** `results/anomalib_protocol/`（新規実行）  
**既存 legacy:** `results/`（削除・上書きしない）

ルール（anomalib 2.2.0 既定の実効 split）:

1. `train/good` を読み込み
2. `test_split_ratio=0.2`, `seed=42` で normal hold-out → test に追加
3. `test/<defect>` 全件を test に含める
4. `val = deepcopy(test)`（`ValSplitMode.SAME_AS_TEST`）
5. ディスク `validation/` は **使用しない**（manifest では `unused_on_disk_validation`）

生成:

```powershell
uv run python -m src.data.build_split_manifest --protocol anomalib --root datasets/ADAM3D
uv run python -m src.data.validate_split_manifest `
  --manifest configs/splits/anomalib_manifest.csv `
  --root datasets/ADAM3D `
  --protocol anomalib `
  --no-strict-leakage
```

**注意:** 同一 part の別 view が train と test に分かれ得る（view leakage の可能性）。`--strict-leakage` では警告ではなく問題として扱う。

### 2. paper protocol (`paper`)

**class:** `paper_protocol`  
**実装状態:** `blocked_on_anomalib_subset`  
**結果ディレクトリ:** `results/paper_protocol/`

論文 Section IV-A（確認済み）:

- train: **すべての** defect-free scans
- defective: part-instance 単位で test:validation = **60:40**
- view-point leakage 防止

公開 anomalib subset だけでは欠陥の part-instance が復元できないため、manifest ビルドは明示エラーとする。  
フルデータセット取得後に再実装する。

### 3. inferred protocol (`inferred`)

**class:** `inferred_protocol`  
**実装状態:** experimental（論文公式ではない）

- 全 good（train+disk validation）→ train
- parseable defect part_id → 60/40 test/val（seed=42）
- unparseable defect → test（part_id 空）

論文値との直接比較には使わない。

## Manifest スキーマ

CSV 列:

`category, part_id, view_id, label, anomaly_type, split, image_path, depth_path, mask_path, stem, parse_status, source_split_dir`

付随:

- `*.meta.json` — seed, categories, status
- `*.sha256` — manifest の SHA-256
- `*.validation.json` — 検証レポート

## 検証項目

- RGB/depth/mask stem 対応
- パス存在
- active split 内の重複なし
- 異常サンプルの mask 欠落（fatal）
- 空 mask（all-zero）は `dataset_quality_issues`（既定では ok を落とさない。`--fail-on-empty-masks` で fatal）
- test に normal/abnormal（protocol 依存）
- part_id leakage（paper/inferred は abnormal の val↔test を strict）
- カテゴリ別件数
- manifest SHA-256

実測（anomalib subset）: 空 mask が複数カテゴリに存在（例: `4m1/under_extrusion`, `gripper_open/cut`, `rackgear/under_extrusion` 等）。
