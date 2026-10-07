# Third-party 3D / RGB+3D baselines

Vendored and lightly patched sources for 3D-ADAM reproduction.

| Directory | Upstream | Role |
|-----------|----------|------|
| `3D-ADAMBench/TransFusion_3D-ADAM` | [PaulMcHard/3D-ADAMBench](https://github.com/PaulMcHard/3D-ADAMBench) | TransFusion RGB+D on ADAM3D |
| `AnomalyDetection-M3DM` | [TencentYoutuResearch/AnomalyDetection-M3DM](https://github.com/TencentYoutuResearch/AnomalyDetection-M3DM) | M3DM hybrid fusion |
| `Real3D-AD` | [M-3LAB/Real3D-AD](https://github.com/M-3LAB/Real3D-AD) | Reg3D-AD (+ PCD benchmark utils) |

Local patches:

- TransFusion `dataset/Dataset.py`: anomalib ADAM3D `rgb/xyz/ground_truth` layout
- TransFusion `Experiment.py`: category list fix, Windows `num_workers=0`
- M3DM `dataset.py` / `main.py`: `adam3d` dataset type + `ground_truth/`
- Real3D-AD `dataset_pc.py` / `main_reg3dad.py`: ADAM3D PCD mode via env vars

Run from repo root via `src/run_*.py` (see root README §15).
