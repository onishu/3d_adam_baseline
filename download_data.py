import argparse
from anomalib.data import ADAM3D

p = argparse.ArgumentParser()
p.add_argument("--root", default="datasets/ADAM3D")
a = p.parse_args()
ADAM3D(root=a.root, category="1m1", num_workers=0).prepare_data()
print("Dataset preparation completed:", a.root)
