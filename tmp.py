import random
import cv2
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "dataset"

IMG_DIR = dataset_dir / "images"
LAB_DIR = dataset_dir / "labels"
OUT_DIR = project_root / "check"

CLASSES = ["slot","mach_nho","mach_lon","usb_to_jtag","day_black",
           "day_lgbt","day_white","pack_circut","day_gray"]

os.makedirs(OUT_DIR, exist_ok=True)

def load_label(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue

            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) == 8:
                xs = coords[0::2]
                ys = coords[1::2]
                boxes.append({
                    "cls": cls,
                    "xs": xs,
                    "ys": ys
                })

    return boxes

def draw_obb(image, xs, ys, color, label):
    pts = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in zip(xs, ys)]
    for i in range(4):
        cv2.line(image, pts[i], pts[(i+1) % 4], color, 2)

    cv2.putText(image, label, (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_split(split):
    print(f"=== Checking {split} ===")

    out_split = OUT_DIR / split
    os.makedirs(out_split, exist_ok=True)

    img_folder = IMG_DIR / split
    lab_folder = LAB_DIR / split

    img_paths = list(img_folder.glob("*.jpg"))
    img_paths += list(img_folder.glob("*.png"))
    img_paths += list(img_folder.glob("*.jpeg"))

    if len(img_paths) == 0:
        print(f"[WARN] Không tìm thấy ảnh trong {img_folder}")
        return

    chosen = random.sample(img_paths, min(10, len(img_paths)))

    for img_path in chosen:
        name = img_path.stem
        lab_path = lab_folder / f"{name}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERR] Không mở được ảnh: {img_path}")
            continue

        boxes = load_label(lab_path)

        for b in boxes:
            color = tuple(int(x) for x in (random.random()*255,
                                           random.random()*255,
                                           random.random()*255))
            label = CLASSES[b["cls"]]
            draw_obb(img, b["xs"], b["ys"], color, label)

        out_path = out_split / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n=== DONE ===")
