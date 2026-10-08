# Source Inventory

外部コード・データ・重みの出所記録。取り込み時は URL / commit SHA / ライセンス / ローカル差分を更新すること。

## Datasets

| 名前 | URL | ライセンス | ローカルパス | 備考 |
|------|-----|------------|--------------|------|
| 3D-ADAM anomalib subset | https://huggingface.co/datasets/pmchard/3D-ADAM_anomalib | CC BY-NC-SA 4.0 | `datasets/ADAM3D` | ZIP hash (anomalib): `ffc4c52afa1566a4165c42300c21758ec8292ff04305c65e81e830abb8200c36` |
| 3D-ADAM full | https://huggingface.co/datasets/pmchard/3D-ADAM | CC BY-NC-SA 4.0 | 未取得 | ~669 GB, part-instance separated |
| DTD | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | 上流ライセンスに従う | `datasets/dtd` | dtd-r1.0.1 展開済（5640 jpg）。path-manifest SHA は smoke JSON / adapter `validate_dtd` |

## Author / upstream repositories

| ローカル | Upstream URL | Commit SHA | ライセンス | ローカル変更 |
|----------|--------------|------------|------------|--------------|
| `third_party/3D-ADAMBench` | https://github.com/PaulMcHard/3D-ADAMBench | `58c7e108a4b1de5789486fd2917245a1907234bf` (docs pin; tree NOGIT) | MIT | ADAM3D layout, Windows num_workers。**runtime adapter** が `get_plane_mask` を ADAM3D-safe に差し替え（third_party 本体は非編集） |
| (参照) | https://github.com/maticfuc/ECCV_TransFusion | `a419dd93f26fa3c140fadf6d35fdf1a79074a515` | MIT | 原著 |
| `third_party/AnomalyDetection-M3DM` | https://github.com/TencentYoutuResearch/AnomalyDetection-M3DM | **未固定 (NOGIT)** | 上流確認要 | adam3d + ground_truth |
| `third_party/Real3D-AD` | https://github.com/M-3LAB/Real3D-AD | **未固定 (NOGIT)** | 上流確認要 | ADAM3D PCD env 対応 |
| (未同梱) | https://github.com/PaulMcHard/3D-ADAM_anomalib | 未取得 | — | 著者 anomalib baseline |
| (未同梱) | https://github.com/PaulMcHard/Anomalib_3D-ADAM | 未取得 | — | 著者 CFA スクリプト |
| (optional) | https://github.com/VitjanZ/3DSR | `6f83317837c6615f893f3107235c75edda90181a` | Apache-2.0 | Phase 5 (CPU adapter; GPU later) |
| (未同梱) | https://github.com/hustCYQ/GLFM-Multi-class-3DAD | 未取得 | — | Phase 6 候補 |
| (未同梱) | https://github.com/PaulMcHard/GLFM | 未取得 | — | Phase 6 候補 |

## Libraries (locked)

| パッケージ | バージョン | 出典 |
|------------|-----------|------|
| anomalib | 2.2.0 | `pyproject.toml` / `uv.lock` |
| torch | 2.8.0+cu126 | pytorch-cu126 index |
| torchvision | 0.23.0+cu126 | 同上 |
| pandas | 2.2.3 | anomalib ADAM3D split 互換のため固定 |

`uv.lock` SHA-256 は `src.common.capture_environment` 実行時に記録する。

## Pretrained weights

| 用途 | パス / URL | SHA-256 | 状態 |
|------|------------|---------|------|
| ImageNet backbones (anomalib) | torch/timm 経由 | 実行時記録 | 自動 |
| PointMAE (M3DM) | `third_party/AnomalyDetection-M3DM/checkpoints/` | 未記録 | 要配置 |
| 3DSR DADA_D / DADA_RGB_D | Phase 5 | `scripts/download_3dsr_weights.ps1` | 手動DL |

## Paper

- arXiv: https://arxiv.org/abs/2507.07838
- Metrics registry: `configs/paper_metrics.yaml`
