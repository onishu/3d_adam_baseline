# 3D-ADAM anomalib baseline reproduction

3D-ADAM 論文  
> *“3D-ADAM: A Dataset for 3D Anomaly Detection in Additive Manufacturing”* [arXiv:2507.07838][web:25]  

に掲載されている anomalib ベースライン (PatchCore / PaDiM / CFA) を、  
**Windows + PowerShell + uv + RTX 3060** の環境で再現するためのプロジェクトです。

- データセット: 3D-ADAM anomalib subset (MechMind Nano 単眼 RGB+XYZ)[web:26]
- ライブラリ: anomalib 2.2.0[web:93]
- ベースラインコード参考:
  - 著者 anomalib ベースライン: `PaulMcHard/3D-ADAM_anomalib` (CFA/PatchCore/PaDiM 実装)[web:208][web:214]
  - 著者 CFA スクリプト: `PaulMcHard/Anomalib_3D-ADAM/cfa3D_adam.py`[web:208]

---

## 1. 環境

### 1.1 前提

- OS: Windows 10/11
- シェル: PowerShell
- Python: 3.11
- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- パッケージ管理: [uv](https://github.com/astral-sh/uv)

### 1.2 プロジェクト依存

`pyproject.toml` で以下を固定しています。

- `anomalib==2.2.0`
- `torch==2.8.0+cu126`
- `torchvision==0.23.0+cu126`
- `pandas==2.2.3`  ← 重要 (後述)

---

## 2. セットアップ

### 2.1 依存パッケージインストール

```powershell
uv sync

uv run python env_check.py
```

`env_check.py` は次を表示します。

- anomalib バージョン
- torch バージョンと CUDA ランタイム
- `torch.cuda.is_available()` と GPU 名・VRAM

例:

```text
anomalib: 2.2.0
torch: 2.8.0+cu126
CUDA runtime: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
VRAM GiB: 11.9995...
```

### 2.2 pandas 2.2.3 固定について

anomalib 2.2.0 の ADAM3D datamodule は、Pandas 3.x 環境では `train/val/test` の split が 0 件になり、  
`Zero subset length encountered during splitting` という警告とともにデータが空になります。

そのため、このプロジェクトでは **`pandas==2.2.3` を必須**としています。

---

## 3. データセット

### 3.1 3D-ADAM anomalib subset

- データ提供: Hugging Face `pmchard/3D-ADAM_anomalib`[web:26]
- ZIP サイズ: 約 5.2 GB
- 展開後推奨空き容量: 20 GB 以上

既に `datasets/ADAM3D` がある場合は再ダウンロード不要です。

### 3.2 ダウンロード

```powershell
uv run python download_data.py --root datasets/ADAM3D
```

`download_data.py` は anomalib の `ADAM3D` datamodule を利用して ZIP を取得し、  
カテゴリごとの `train/good`, `validation/good`, `test/<defect>` を展開します。

---

## 4. データ検証

23カテゴリがすべて正しく読み込めるかを確認します。

```powershell
uv run python -m src.verify_dataset --root datasets/ADAM3D
```

期待されるカテゴリ一覧:

```text
1m1, 1m2, 1m3,
2m1, 2m2h, 2m2m,
3m1, 3m2, 3m2c,
4m1, 4m2, 4m2c,
gripper_closed, gripper_open,
helicalgear1, helicalgear2,
rackgear, spiralgear, spurgear,
tapa2m1, tapa3m1, tapa4m1, tapatbb
```

`src.verify_dataset` は各カテゴリについて:

- train / val / test のサンプル数
- RGB `image.shape == (3, 1024, 1024)`
- depth `depth_map.shape == (3, 1024, 1024)`
- mask `gt_mask.shape == (1024, 1024)`

をチェックし、全 23 カテゴリが `OK` になると終了します。

---

## 5. 実験実行

### 5.1 コマンド概要

全モデル共通で `src/run_all.py` を使用します。

```powershell
uv run python -m src.run_all --model <model> [options]
```

- `--model`: `patchcore` | `padim` | `cfa`
- `--root`: データルート (既定 `datasets/ADAM3D`)
- `--categories`: 限定カテゴリ (未指定なら 23カテゴリすべて)
- `--train-batch-size`: 学習時バッチサイズ
- `--eval-batch-size`: 評価時バッチサイズ
- `--num-workers`: DataLoader の `num_workers`
- `--out`: 結果出力ルート (既定 `results`)
- `--force`: 既存 JSON を上書きして再計算

1カテゴリごとに `results/raw/<model>/<category>.json` を生成し、完了済みカテゴリは次回以降 `SKIP` されます。

---

## 6. Smoke test

PatchCore を `1m1` のみで試します。

```powershell
uv run python -m src.run_all `
    --model patchcore `
    --categories 1m1 `
    --train-batch-size 16 `
    --eval-batch-size 16 `
    --num-workers 0
```

GPU メモリ不足が出た場合:

```powershell
uv run python -m src.run_all `
    --model patchcore `
    --categories 1m1 `
    --train-batch-size 8 `
    --eval-batch-size 8 `
    --num-workers 0
```

---

## 7. 23カテゴリ一括実行

### 7.1 標準条件 (再現プロジェクトの既定)

```powershell
# PatchCore
uv run python -m src.run_all --model patchcore

# PaDiM
uv run python -m src.run_all --model padim

# CFA
uv run python -m src.run_all --model cfa
```

ログをファイル保存しつつコンソールでも確認する場合:

```powershell
mkdir logs -Force

uv run python -m src.run_all --model padim 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath logs\padim.log

uv run python -m src.run_all --model cfa 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath logs\cfa.log
```

---

## 8. 著者寄せ CFA 実験 (batch 32)

CFA だけは、著者公開スクリプトに近い条件として **batch size 32** で別ディレクトリへ出力します。[web:208][web:214]

```powershell
uv run python -m src.run_all `
    --model cfa `
    --train-batch-size 32 `
    --eval-batch-size 32 `
    --num-workers 0 `
    --out results_author_cfa 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath logs\cfa-author-settings.log
```

- VRAM 12 GB だと batch 32 は問題なく動作する想定です。
- OOM が出る場合は batch 16 に戻し、「著者条件からの差」として README に明記してください。

---

## 9. 結果集計

`src/compare.py` が per-category CSV と平均値を生成します。

### 9.1 標準結果

```powershell
uv run python -m src.compare --model patchcore
uv run python -m src.compare --model padim
uv run python -m src.compare --model cfa
```

### 9.2 著者寄せ CFA 結果

```powershell
uv run python -m src.compare `
    --model cfa `
    --out results_author_cfa
```

生成物:

```text
results/
  raw/<model>/_env.json
  raw/<model>/<category>.json
  per_category_<model>.csv
  summary_<model>.csv
  summary_<model>.json

results_author_cfa/
  raw/cfa/_env.json
  raw/cfa/<category>.json
  per_category_cfa.csv
  summary_cfa.csv
  summary_cfa.json
```

---

## 10. 再現結果 (Reproduction results)

### 10.1 実験環境

- Windows 11 + PowerShell + uv
- GPU: RTX 3060 12 GB
- Python: 3.11
- anomalib: 2.2.0[web:93]
- torch: 2.8.0+cu126
- pandas: 2.2.3
- seed: 42
- ADAM3D datamodule: 全 23カテゴリで検証 OK

### 10.2 PatchCore (23/23 categories)

標準条件 (batch 16, out=`results`):

| metric        | reproduced (%) | paper Table V (%) | delta (pt) |
|---------------|---------------:|-------------------:|----------:|
| image_AUROC   | 96.81          | 97.60              | -0.79     |
| image_F1Score | 96.97          | 97.70              | -0.73     |
| pixel_AUROC   | 87.13          | 84.20              | +2.93     |
| pixel_F1Score | 10.21          | 6.10               | +4.11     |

画像レベル 2 指標が 1pt 未満の差であり、  
PatchCore の anomalib ベースラインはほぼ完全に再現できていると判断します。

### 10.3 PaDiM (23/23 categories)

標準条件 (batch 16, out=`results`):

| metric        | reproduced (%) | paper Table V (%) | delta (pt) |
|---------------|---------------:|-------------------:|----------:|
| image_AUROC   | 95.48          | 96.40              | -0.92     |
| image_F1Score | 97.02          | 97.50              | -0.48     |
| pixel_AUROC   | 87.83          | 84.60              | +3.23     |
| pixel_F1Score | 6.51           | 4.10               | +2.41     |

PatchCore 同様、画像レベル 2 指標が 1pt 前後で、  
PaDiM の anomalib ベースラインも再現成功と判断できます。

### 10.4 CFA (23/23 categories, author-like settings)

著者寄せ CFA 実験 (batch 32, out=`results_author_cfa`) の結果:

| metric        | reproduced (%) | paper Table V (%) | delta (pt) |
|---------------|---------------:|-------------------:|----------:|
| image_AUROC   | 76.97          | 79.40              | -2.43     |
| image_F1Score | 94.00          | 93.90              | +0.10     |
| pixel_AUROC   | 85.69          | 87.00              | -1.31     |
| pixel_F1Score | 6.28           | 5.80               | +0.48     |

Image AUROC で約 -2.4pt、Pixel AUROC で約 -1.3pt の差がありますが、  
論文値と同傾向であり、CFA ベースラインも **「ほぼ再現」** と見なせる範囲です。

---

## 11. Pixel F1 の扱いについて

Pixel F1 は異常マップを二値化する閾値に強く依存するため、  
AUROC と比べて環境や実装の違いによる揺らぎが大きくなります。[web:232]

anomalib 2.x の既定では:

- Anomaly score を [0,1] に正規化
- Precision–Recall 上で F1 を最大化する adaptive threshold を用いる

ADAM3D datamoduleでは `val_split_mode=SAME_AS_TEST` のため、validation が test の deep copy になり、  
test の F1 が最大化される閾値が選ばれやすく、Pixel F1 が論文値より高めに出る傾向があります。

そのため、本プロジェクトでは:

- AUROC (閾値非依存) と F1 (閾値依存) を分けて評価する
- Pixel F1 は「同傾向かどうか」を見る補助指標として扱う
- 3 モデルとも Pixel F1 は論文値よりやや高めだが、方向とオーダーは一致している

としています。

---

## 12. 再現判定の指針

- ビット単位一致ではなく、**指標の「同傾向・平均値の近さ」**で判定する
- 特に Image AUROC / Image F1 を重視してベースラインを評価する
- Pixel AUROC / Pixel F1 は閾値依存性を踏まえて補助的に見る
- `results/raw/<model>/_env.json` に実行環境が保存されており、いつでも監査可能
- 著者環境 (`anomalib==2.1.0`) との厳密な差分は README に明示する

---

## 13. 実験環境の記録 (推奨)

Git 管理と合わせて、次のように実験環境を保存しておくと便利です。

```powershell
uv pip freeze | Out-File -Encoding utf8 results\environment-<model>.txt

git rev-parse HEAD 2>$null |
    Out-File -Encoding utf8 results\git-commit.txt
```

---

## 14. 既知の注意点

1. **Pandas 3.x の場合**  
   - ADAM3D datamodule の split 比較が Enum vs string になり、train/val/test が 0 件になる  
   - 本プロジェクトは `pandas==2.2.3` 固定で回避

2. **CFA + ADAM3D + anomalib 2.2.0 の組み合わせ**  
   - 一部の設定では TorchMetrics 側の shape mismatch  
     `Predictions and targets are expected to have the same shape, but got torch.Size([]) and torch.Size([1])`  
     が出ることがあり、1カテゴリだけ評価失敗する場合がある[web:235]  
   - 著者寄せ設定 (batch 32, out=`results_author_cfa`) では 23/23 カテゴリ完走を確認済み

---

## 15. 今後の拡張

- TransFusion / RGB+3D fusion の再現 (著者の `3D-ADAMBench` リポジトリ)[web:152]
- 別 GPU / 別 OS での再現性検証
- 評価指標を増やした詳細解析 (per-category の分布、カテゴリごとの難易度分析など)
- 3D 異常検知手法 (Reg3D-AD、M3DM 等) との比較実験[web:206]

---

この README は、3D-ADAM 論文の anomalib ベースラインを  
**Windows + RTX 3060** 環境で再現するための手順と、得られた結果を記録することを目的としています。