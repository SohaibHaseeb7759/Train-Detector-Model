"""
Convert CVAT polygon annotations (annotations.xml) into YOLO segmentation
format and build a train/val dataset ready for `yolo segment train`.

Usage:
    python prepare_dataset.py --images <folder-with-jpgs> --val-split 0.2

The script:
  1. Parses annotations.xml (CVAT 1.1 format, polygon labels).
  2. Writes one YOLO .txt label per image (normalized polygon points).
  3. Copies images + labels into dataset/images/{train,val} and
     dataset/labels/{train,val}.
  4. Generates data.yaml.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def parse_annotations(xml_path):
    """Return (labels, images) where images is a list of dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect label names in the order CVAT declares them -> class ids.
    label_names = [lbl.findtext("name").strip()
                   for lbl in root.findall(".//labels/label")]
    label_to_id = {name: i for i, name in enumerate(label_names)}

    images = []
    for img in root.findall("image"):
        name = img.get("name")
        width = float(img.get("width"))
        height = float(img.get("height"))
        polygons = []
        for poly in img.findall("polygon"):
            label = poly.get("label").strip()
            cls_id = label_to_id[label]
            pts = []
            for pair in poly.get("points").split(";"):
                x, y = pair.split(",")
                # normalize to 0..1 and clamp
                nx = min(max(float(x) / width, 0.0), 1.0)
                ny = min(max(float(y) / height, 0.0), 1.0)
                pts.extend([nx, ny])
            polygons.append((cls_id, pts))
        images.append({"name": name, "polygons": polygons})

    return label_names, images


def write_label_file(dst_txt, polygons):
    lines = []
    for cls_id, pts in polygons:
        coords = " ".join(f"{v:.6f}" for v in pts)
        lines.append(f"{cls_id} {coords}")
    dst_txt.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(PROJECT / "annotations.xml"))
    ap.add_argument("--images", default=str(PROJECT / "train_images"),
                    help="Folder containing the .jpg image files.")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir = Path(args.images)
    label_names, images = parse_annotations(args.xml)
    print(f"Labels: {label_names}")
    print(f"Images in XML: {len(images)}")

    # Verify image files exist.
    found, missing = [], []
    for img in images:
        if (images_dir / img["name"]).exists():
            found.append(img)
        else:
            missing.append(img["name"])
    if missing:
        print(f"WARNING: {len(missing)} image(s) referenced in XML not found "
              f"in {images_dir}. First few: {missing[:5]}")
    print(f"Images found on disk: {len(found)}")
    if not found:
        raise SystemExit("No images found. Point --images at the right folder.")

    # Train/val split.
    random.seed(args.seed)
    random.shuffle(found)
    n_val = max(1, int(len(found) * args.val_split))
    val_set = set(id(x) for x in found[:n_val])

    ds = PROJECT / "dataset"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = 0
    for img in found:
        split = "val" if id(img) in val_set else "train"
        if split == "train":
            n_train += 1
        # copy image
        shutil.copy2(images_dir / img["name"],
                     ds / "images" / split / img["name"])
        # write label (same stem, .txt). Empty file if no polygons.
        stem = Path(img["name"]).stem
        write_label_file(ds / "labels" / split / f"{stem}.txt",
                         img["polygons"])

    # data.yaml
    yaml_text = (
        f"path: {ds.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(label_names))
    )
    (PROJECT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"\nDone. train={n_train}, val={len(found) - n_train}")
    print(f"data.yaml written to {PROJECT / 'data.yaml'}")


if __name__ == "__main__":
    main()
