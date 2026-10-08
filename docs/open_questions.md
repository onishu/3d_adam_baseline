# Open Questions

分類: **未確認** / **仮定** / **独自設定**

## データ分割

| ID | 事項 | 分類 | 現状 |
|----|------|------|------|
| Q-SPLIT-01 | 論文の part-instance ID が anomalib subset のどのフィールドに対応するか | 未確認 | 欠陥の多くは `000.png` のみで復元不可 |
| Q-SPLIT-02 | good の `{NNN}_{A\|B\|C\|D}` で NNN=part, letter=view か | 仮定 | 命名規則からの推定。sensor ID の可能性も排除しきれない |
| Q-SPLIT-03 | ディスク `validation/good`（多くは `*_A`）の公式意味 | 未確認 | anomalib 既定では未使用 |
| Q-SPLIT-04 | 論文 60:40 は test:val か val:test か | 未確認 | 本文は "test and validation set in a 60:40 share"（test が先）と読む |
| Q-SPLIT-05 | Table IV と Table V で同一 split か | 未確認 | 本文は同一 protocol 記述の後に両表を提示 |
| Q-SPLIT-06 | フル HF データセット上の公式 split ファイルの有無 | 未確認 | HF 説明は part-instance 分離のみ明記 |

## 指標

| ID | 事項 | 分類 | 現状 |
|----|------|------|------|
| Q-MET-01 | 論文 F1 の閾値選択セット（val vs test） | 未確認 | anomalib SAME_AS_TEST は test 最適化になりやすい |
| Q-MET-02 | 論文 AUPRO の FPR 上限・実装 | 未確認 | MVTec 慣例 0.3 を project default とするが論文未記載 |
| Q-MET-03 | TransFusion `PRO Img` が論文 AUPRO と同一定義か | 未確認 | 作者コード出力を採用予定だが差分調査が必要 |
| Q-MET-04 | macro average の重み（カテゴリ均等） | 仮定 | 論文はカテゴリ列の平均と解釈 |

## モデル条件

| ID | 事項 | 分類 | 現状 |
|----|------|------|------|
| Q-MDL-01 | 著者 anomalib バージョン（2.1.0?）と本リポ 2.2.0 の差 | 未確認 | README に言及あり |
| Q-MDL-02 | Table V PatchCore が RGB か depth/XYZ か | 未確認 | 現行は anomalib Depth datamodule + 2D RGB モデル |
| Q-MDL-03 | Table IV PatchCore と Table V PatchCore の入力差 | 未確認 | 論文値も異なる |
| Q-MDL-04 | UniNet / DinoMaly の著者実行設定 | 未確認 | Phase 7 |
| Q-MDL-05 | 3DSR が depth-only か RGB+Depth か（Table V） | 未確認 | Phase 5 |
| Q-MDL-06 | GLFM single-class vs multi-class | 未確認 | Phase 6 |
| Q-MDL-07 | TransFusion `loss=nan` | **解決（主因特定）** | ADAM3D 角 Z=0 により upstream `get_plane_mask` が全ゼロ foreground → 合成異常が空。DTD 不完全も併発。adapter で valid-point plane fit + fail-fast |
| Q-MDL-08 | 著者は ADAM3D に別 foreground mask / 別 plane thr を使ったか | 未確認 | `--no-fg-masks` が既定。`fg_masks/adam3d_masks` ローカル未同梱（0 files） |
| Q-MDL-11 | 本学習ループを新entrypointへ完全移植したか | 独自設定 | Phase 4 は smoke のみ。`--full-train` は拒否。承認後に adapter 付き full train を実装/解禁する |
| Q-MDL-09 | upstream Resize((width,height)) の (h,w) 取り違えを著者が意図したか | 未確認 | RGB/depth は互いに整合して (C,320,256) |
| Q-MDL-10 | 空 abnormal GT mask の公式扱い | 未確認 | ヒューリスティック分類のみ（`anomalous_but_not_visible` 等） |

## データセット範囲

| ID | 事項 | 分類 | 現状 |
|----|------|------|------|
| Q-DATA-01 | 論文 28 categories vs anomalib subset 23 categories | 未確認 | subset=MechMind Nano 由来と README 記載 |
| Q-DATA-02 | 論文数値が 23-cat subset 平均か 28-cat / 複数センサ平均か | 未確認 | 著者 anomalib リポは 23 cat |

## 独自設定（論文非記載・プロジェクト都合）

| ID | 事項 | 分類 |
|----|------|------|
| P-OWN-01 | seed=42 を既定（TransFusion 入口の default seed=3 は作者寄り） | 独自設定 |
| P-OWN-02 | Lightning Trainer 直叩き（symlink 回避） | 独自設定 |
| P-OWN-03 | pandas==2.2.3 固定 | 独自設定（anomalib 互換） |
| P-OWN-04 | 再現判定閾値 ±1 / ±3 pt | 独自設定（論文公式ではない） |
| P-OWN-05 | inferred protocol | 独自設定 |
