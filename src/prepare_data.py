"""
prepare_data.py
----------------
Merges THREE source datasets into one unified 6-class folder structure,
then splits into train/val/test (70/15/15, stratified per class).

Source 1 -- Kaggle "Student-engagement-dataset":
    Engaged/Focused, Engaged/Confused, Engaged/Frustrated
    Not Engaged/Bored, Not Engaged/Drowsy, Not Engaged/Looking Away

Source 2 -- custom "Maleha" dataset:
    Crop_images/Attentive   -> Focused
    Crop_images/Bored       -> Bored
    Crop_images/Confused    -> Confused
    Crop_images/Distracted  -> Looking Away

Source 3 -- custom "Zoha" dataset (folder names lowercase, some .heic
iPhone photos which are auto-converted to .jpg):
    attentive   -> Focused
    bored       -> Bored
    confused    -> Confused
    Distracted  -> Looking Away

Final unified classes (6):
    Bored, Confused, Drowsy, Focused, Frustrated, Looking Away

Run (defaults match this project's own folder layout, so plain
`python src/prepare_data.py` works if you dropped your raw folders in
data/sources/<kaggle|maleha|zoha>):

    python src/prepare_data.py \
        --kaggle_dir data/sources/kaggle \
        --maleha_dir data/sources/maleha \
        --zoha_dir   data/sources/zoha \
        --out_dir    data/prepared
"""
import argparse
import random
import shutil
from pathlib import Path

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from PIL import Image

CLASSES = ["Bored", "Confused", "Drowsy", "Focused", "Frustrated", "Looking Away"]

KAGGLE_MAP = {
    "Focused": "Focused", "Confused": "Confused", "Frustrated": "Frustrated",
    "Bored": "Bored", "Drowsy": "Drowsy", "Looking Away": "Looking Away",
}
# Custom datasets (Maleha / Zoha) share the same 4-class scheme; matched
# case-insensitively since Zoha's folders are lowercase.
CUSTOM_MAP = {
    "attentive": "Focused", "bored": "Bored",
    "confused": "Confused", "distracted": "Looking Away",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def collect_kaggle(root: Path):
    buckets = {c: [] for c in CLASSES}
    if not root or not root.exists():
        return buckets
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMG_EXTS or "__MACOSX" in path.parts:
            continue
        label = path.parent.name
        if label in KAGGLE_MAP:
            buckets[KAGGLE_MAP[label]].append(path)
    return buckets


def collect_custom(root: Path):
    buckets = {c: [] for c in CLASSES}
    if not root or not root.exists():
        return buckets
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMG_EXTS or "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        label = path.parent.name.lower()
        if label in CUSTOM_MAP:
            buckets[CUSTOM_MAP[label]].append(path)
    return buckets


def split_list(items, train=0.7, val=0.15, seed=42):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def save_as_jpg(src: Path, dst: Path):
    """Opens any supported image (incl. HEIC) and saves it as a JPEG."""
    im = Image.open(src).convert("RGB")
    im.save(dst, "JPEG", quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", default=None, help="Path to Student-engagement-dataset root")
    ap.add_argument("--maleha_dir", default=None, help="Path to Maleha/Crop_images root")
    ap.add_argument("--zoha_dir", default=None, help="Path to Zoha root")
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    kaggle_buckets = collect_kaggle(Path(args.kaggle_dir)) if args.kaggle_dir else {c: [] for c in CLASSES}
    maleha_buckets = collect_custom(Path(args.maleha_dir)) if args.maleha_dir else {c: [] for c in CLASSES}
    zoha_buckets = collect_custom(Path(args.zoha_dir)) if args.zoha_dir else {c: [] for c in CLASSES}

    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)

    summary = {}
    for cls in CLASSES:
        combined = kaggle_buckets[cls] + maleha_buckets[cls] + zoha_buckets[cls]
        train, val, test = split_list(combined, seed=args.seed)
        summary[cls] = {
            "kaggle": len(kaggle_buckets[cls]), "maleha": len(maleha_buckets[cls]),
            "zoha": len(zoha_buckets[cls]), "total": len(combined),
            "train": len(train), "val": len(val), "test": len(test),
        }
        for split_name, split_files in (("train", train), ("val", val), ("test", test)):
            dest_dir = out_root / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(split_files):
                dst = dest_dir / f"{src.stem}_{i:05d}.jpg"
                try:
                    save_as_jpg(src, dst)
                except Exception as e:
                    print(f"  skip {src} ({e})")

    print(f"{'Class':<15}{'Kaggle':>8}{'Maleha':>8}{'Zoha':>8}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}")
    for cls in CLASSES:
        s = summary[cls]
        print(f"{cls:<15}{s['kaggle']:>8}{s['maleha']:>8}{s['zoha']:>8}{s['total']:>8}{s['train']:>8}{s['val']:>8}{s['test']:>8}")
    print(f"\nDone. Prepared dataset written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
