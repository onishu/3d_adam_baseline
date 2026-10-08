# 3D-ADAM Reproduction Audit (Phase 0)

Date: 2026-10-08  
Repo commit at audit: `df9150782a91e685ffe4998f539a02267dca4323`  
Legacy results git-commit.txt: `5f2b9801de10db30a7502c8cbb480399cdb74213`

## 現在の実装一覧

| 入口 | 役割 | 出力 | 状態 |
|------|------|------|------|
| `src.verify_dataset` | 23カテゴリ読み込み検証 | stdout | 実装済 |
| `src.run_all` | PatchCore / PaDiM / CFA (Lightning Trainer) | `results/raw/<model>/` | 実装済・実行済 |
| `src.compare` | per-category / macro summary | `results/summary_*` | 実装済 |
| `src.run_transfusion` | TransFusion RGB+D | `results_transfusion/` | 実装済・学習実行中 |
| `src.run_rgbd_fusion` | RGB/Depth PatchCore score fusion | `results_rgbd_fusion/` | 実装済 |
| `src.run_m3dm` | M3DM | `results_m3dm/` | 入口あり |
| `src.run_reg3dad` / `src.export_adam3d_pcd` | Reg3D-AD | `results_reg3dad/` | 入口あり |
| `src.smoke_3d` | 3D adapter smoke | stdout | 実装済 |
| `src.common.*` | 共通基盤 (Phase 2) | — | 新規 |
| `src.data.build_split_manifest` | split manifest (Phase 1) | `configs/splits/` | 新規 |

## Table IV / Table V / 論文外拡張の分類

### Table IV（2D）— 論文 Section IV / Table IV

| モデル | 論文値 (pct) | 本リポ |
|--------|-------------:|--------|
| PatchCore (`patchcore_2d`) | 96.2 / 95.9 / 86.5 / 6.9 | **未実装**（既存 PatchCore は Table V 条件） |
| UniNet | 92.4 / 92.3 / 89.9 / 5.3 | **未実装** |
| DinoMaly | 83.4 / 90.3 / 95.9 / 20.0 | **未実装** |

### Table V（3D / RGB+3D）

| モデル | 論文値 (主要) | 本リポ |
|--------|---------------|--------|
| CFA | I-AUROC 79.4 | `results` (22/23) + `results_author_cfa` (23/23, batch32) |
| PaDiM | 96.4 | `results` 23/23 |
| PatchCore | 97.6 | `results` 23/23 |
| 3DSR | 48.5 / 49.4 / AUPRO 18.4 | **未実装** |
| TransFusion | 63.3 / 64.4 / AUPRO 31.0 | 入口あり・部分実行 |
| GLFM | 79.1 / 82.0 / AUPRO 60.3 | **未実装** |

### 論文外拡張（extensions）

- RGB+Depth PatchCore fusion (`src.run_rgbd_fusion`)
- M3DM (`src.run_m3dm`)
- Reg3D-AD (`src.run_reg3dad`)

これらは Table V 直接比較に混ぜない。

## 各モデルの実装元

| モデル | 実装元 | 備考 |
|--------|--------|------|
| PatchCore / PaDiM / CFA | anomalib 2.2.0 + `src.models.build_model` | 著者は anomalib 2.1.0 系を使用した可能性（未確認） |
| TransFusion | `third_party/3D-ADAMBench/TransFusion_3D-ADAM` (PaulMcHard/3D-ADAMBench) | NOGIT（commit SHA 未固定） |
| M3DM | `third_party/AnomalyDetection-M3DM` | NOGIT |
| Reg3D-AD | `third_party/Real3D-AD` | NOGIT |
| 3DSR / GLFM | 未同梱 | Phase 5/6 |

## 使用中のデータ split

### official anomalib protocol（現行 `src.run_all`）

出典: `anomalib.data.datamodules.depth.adam_3d.ADAM3D` 既定値

- `TestSplitMode.FROM_DIR` + `test_split_ratio=0.2` + `seed=42`
- test に normal が無いため、`train/good` から 20% を hold-out して test に追加
- `ValSplitMode.SAME_AS_TEST` → **validation は test の deep copy**
- ディスク上の `validation/` ディレクトリは **読み込まれない**
- 1m1 実測例: train=39, val=49, test=49（defect 40 + hold-out good 9）

### paper protocol（論文 Section IV-A）

確認済み文言:

> anomaly-free training set, containing all of the defect-free scans  
> annotated defective scans partitioned into a test and validation set in a 60:40 share  
> Partitioning is performed on a part-instance level such that view-point leakage is prevented

公開 anomalib subset では欠陥ファイルの多くが `000.png` 形式で **part-instance ID が復元できない** → paper protocol は **blocked**。

### inferred protocol

anomalib subset 上の最良努力。論文公式ではない。

## 使用中の metrics

| 指標 | 現行取得方法 | 注意 |
|------|-------------|------|
| image/pixel AUROC, F1 | anomalib Lightning `trainer.test` のキー | F1 は adaptive threshold |
| AUPRO | TransFusion 出力 CSV (`PRO Img`) | 定義差の監査が Phase 3/4 で必要 |
| 集約 | カテゴリ平均 ×100（macro） | micro は未使用 |

**重要:** `SAME_AS_TEST` のため、F1 閾値選択に test が使われ得る（test-optimized / oracle に近い）。論文比較時は provenance を分離する。

## 既存結果の保存場所

| パス | 内容 |
|------|------|
| `results/raw/{patchcore,padim,cfa}/` | 標準 batch16 |
| `results/summary_*.json/csv`, `per_category_*.csv` | 集計 |
| `results_author_cfa/` | CFA batch32 |
| `results_transfusion/raw/transfusion/` | 1m1, 1m2（時点監査） |

新規 protocol 結果は上書きせず:

```text
results/anomalib_protocol/
results/paper_protocol/
results/author_cfa/
results/extensions/
```

## 各モデル生成引数（現行）

出典: `src/models.py`, `src/run_all.py`, `_env.json`

| モデル | クラス | 主要引数 | epochs | batch | seed | 入力 |
|--------|--------|----------|-------:|------:|-----:|------|
| PatchCore | `anomalib.models.Patchcore` | wide_resnet50_2, layer2/3, coreset=0.1, k=9 | 1 | 16 | 42 | ADAM3D RGB (depth datamoduleだが2DモデルはRGB) |
| PaDiM | `Padim` | resnet18, layer1/2/3 | 1 | 16 | 42 | 同上 |
| CFA | `Cfa` | wide_resnet50_2, gamma_c/d=1, knn=3 | 10 | 16 (author 32) | 42 | 同上 |
| TransFusion | third_party Experiment | mode=rgbd, 320×256, epoch 1501 | 1501 | (作者コード既定) | default 3 | RGB+Depth |

## 再現上のリスク

1. **split 不一致**: 論文 part-instance 60:40 vs anomalib hold-out + SAME_AS_TEST
2. **F1 oracle 化**: val=test により閾値が test 最適化されやすい
3. **Table IV vs V PatchCore 混同**: 論文値も異なる（96.2 vs 97.6）
4. **CFA 標準条件 1m1 失敗**: shape mismatch（author batch32 では 23/23）
5. **TransFusion 実行中 loss=nan**（監査時）— プロセスは停止しない
6. **third_party に git commit 無し**
7. **Windows symlink (WinError 1314)**: Engine 回避のため Trainer 直叩き
8. **既存 results は gitignore** — ローカル再現資産

## 残タスク（優先順位）

1. **P0** 監査ドキュメント ← 本ファイル
2. **P1** split manifest 固定・検証
3. **P2–P3** 共通基盤・metrics
4. **P4** TransFusion 本再現（smoke 後に承認待ち）
5. **P5–P6** 3DSR / GLFM
6. **P7** Table IV
7. **P8–P13** 拡張分離・レポート・定性・環境・最終計画

## 完了条件（プロジェクト全体）

ユーザー指定の「論文ベンチマーク再現完了」条件を満たすこと。  
Phase 0–3 時点では **基盤と protocol 文書化まで**が完了条件。

## 再利用可能部分

- `src.models.build_model` / `CATEGORIES` / Lightning Trainer 回避パターン
- `src.compare` の集約ロジック（schema 拡張が必要）
- `third_party` TransFusion ADAM3D adapter
- 既存 raw JSON（legacy として保持）
