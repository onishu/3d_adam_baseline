import argparse
import sys
from anomalib.data import ADAM3D
from .categories import CATEGORIES

p = argparse.ArgumentParser()
p.add_argument("--root", default="datasets/ADAM3D")
p.add_argument("--categories", nargs="*")
a = p.parse_args()
failed = []
categories = a.categories or CATEGORIES
print(f"{'category':<16}{'train':>7}{'val':>7}{'test':>7}  status")
for category in categories:
    try:
        dm = ADAM3D(root=a.root, category=category, num_workers=0, seed=42)
        dm.prepare_data()
        dm.setup()
        counts = len(dm.train_data), len(dm.val_data), len(dm.test_data)
        item = dm.test_data[0]
        ok = all(counts) and item.image.ndim == 3 and item.depth_map is not None
        print(f"{category:<16}{counts[0]:>7}{counts[1]:>7}{counts[2]:>7}  {'OK' if ok else 'BAD'}")
        if not ok:
            failed.append(category)
    except Exception as exc:
        failed.append(category)
        print(f"{category:<16} ERROR {type(exc).__name__}: {exc}")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
print(f"All {len(categories)} categories verified OK.")
