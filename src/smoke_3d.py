"""Quick smoke checks for 3D method adapters (no full training)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF_ROOT = ROOT / "third_party" / "3D-ADAMBench" / "TransFusion_3D-ADAM"


def smoke_transfusion(category: str, root: str) -> None:
    data_path = str(Path(root).resolve())
    dtd_path = str((ROOT / "datasets" / "dtd" / "images").resolve())
    fg_mask_path = str((ROOT / "fg_masks").resolve())
    # Ensure TransFusion package `dataset/` wins over M3DM module `dataset.py`.
    for name in list(sys.modules):
        if name == "dataset" or name.startswith("dataset.") or name.startswith("utils.") or name.startswith("model."):
            del sys.modules[name]
    sys.path = [str(TF_ROOT), *[p for p in sys.path if "AnomalyDetection-M3DM" not in p]]
    os.chdir(TF_ROOT)
    import argparse as ap
    from utils.ParsingUtils import create_model, get_dataset
    import torch

    args = ap.Namespace(
        data_path=data_path,
        dataset="adam3d",
        category=category,
        mode="rgbd",
        img_width=320,
        img_height=256,
        dtd_path=dtd_path,
        no_fg_masks=False,
        fg_mask_path=fg_mask_path,
        seed=3,
        diffusion_num_steps=75,
        unet_channel_num=64,
        unet_ch_mults=[1, 2, 2, 2],
        unet_attn=[False, False, True, True],
        unet_group_conv=[True, True, True, True],
        unet_n_blocks=2,
        lr=1e-5,
        step_size=800,
        gamma=0.1,
    )
    train = get_dataset(args, True)
    batch = train[0]
    x = batch["image"].unsqueeze(0).cuda()
    plane = batch["plane_mask"].unsqueeze(0).cuda()
    net = create_model(args)
    net.train()
    loss = net.loss(x, idx=torch.tensor([0]), plane_mask=plane)
    loss.backward()
    print(f"[transfusion] ok train={len(train)} shape={tuple(x.shape)} loss={float(loss):.4f}")


def smoke_pcd(category: str, root: str) -> None:
    from .export_adam3d_pcd import export_category

    out = ROOT / "datasets" / "ADAM3D_pcd"
    n_train, n_test = export_category(Path(root), out, category, max_points=2000)
    print(f"[pcd] ok train={n_train} test={n_test} out={out / category}")


def smoke_m3dm_dataset(category: str, root: str) -> None:
    sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDetection-M3DM"))
    from dataset import TestDataset, TrainDataset

    train = TrainDataset(category, img_size=224, dataset_path=root)
    test = TestDataset(category, img_size=224, dataset_path=root)
    tr = train[0]
    te = test[0]
    print(
        f"[m3dm] ok train={len(train)} test={len(test)} "
        f"rgb={tuple(tr[0][0].shape)} xyz={tuple(tr[0][1].shape)}"
    )
    print(f"[m3dm] test gt={tuple(te[1].shape)} label={int(te[2])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/ADAM3D")
    p.add_argument("--category", default="1m1")
    p.add_argument(
        "--only",
        nargs="*",
        default=["transfusion", "pcd", "m3dm"],
        choices=["transfusion", "pcd", "m3dm"],
    )
    a = p.parse_args()
    if "pcd" in a.only:
        smoke_pcd(a.category, a.root)
    if "m3dm" in a.only:
        smoke_m3dm_dataset(a.category, a.root)
    if "transfusion" in a.only:
        smoke_transfusion(a.category, a.root)


if __name__ == "__main__":
    main()
