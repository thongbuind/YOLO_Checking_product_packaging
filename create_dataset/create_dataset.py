from pathlib import Path
import yaml
import json
import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

create_dataset_dir = project_root / "create_dataset"
raw_img_dir = create_dataset_dir / "raw_image"
raw_lab_dir = create_dataset_dir / "labels"

dataset_dir = project_root / "dataset"
data_yaml_path = dataset_dir / "data.yaml"
config_file = project_root / "config.json"

with open(config_file, "r") as f:
    config = json.load(f)

image_size = config["image_size"]
train_ratio = config["train_ratio"]
val_ratio = config["val_ratio"]
classes = ["slot", "mach_nho", "mach_lon", "usb_to_jtag",
           "day_black", "day_lgbt", "day_white", "pack_circut", "day_gray"]

print(f"Dataset Configuration:")
print(f"  Image size: {image_size}")
print(f"  Train ratio: {train_ratio}")
print(f"  Val ratio: {val_ratio}")
print(f"  Test ratio: {1 - train_ratio - val_ratio}")

# ---------------------------
# 1. Chuẩn hóa extension
# ---------------------------
def normalize_extensions(folder: Path):
    """Chuyển tất cả extension ảnh về lowercase"""
    count = 0
    for ext in ["*.JPG", "*.JPEG", "*.PNG"]:
        for path in folder.glob(ext):
            new_path = path.with_suffix(path.suffix.lower())
            if path != new_path:
                path.rename(new_path)
                count += 1
    print(f"Normalized {count} file extensions")

normalize_extensions(raw_img_dir)

# ---------------------------
# 2. Đọc label txt (OBB format)
# ---------------------------
def load_label(txt_path):
    """Load labels in OBB format: class x1 y1 x2 y2 x3 y3 x4 y4"""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 9:  # class + 8 coordinates
                continue
            cls = int(parts[0])
            coords = parts[1:]
            boxes.append((cls, coords))
    return boxes

# ---------------------------
# 3. Ghi label
# ---------------------------
def save_label(path, labels):
    """Save labels in OBB format"""
    with open(path, "w") as f:
        for cls, coords in labels:
            f.write(f"{cls} " + " ".join(f"{x:.6f}" for x in coords) + "\n")

# ---------------------------
# 4. Resize ảnh và label (FIX: logic resize)
# ---------------------------
def resize_img_and_label(img, labels, size):
    """
    Resize image và scale coordinates về normalized [0,1]
    """
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (size, size))

    new_labels = []
    for cls, coords in labels:
        # coords = [x1, y1, x2, y2, x3, y3, x4, y4] (8 values)
        # Giả sử coords đã ở dạng normalized [0,1]
        # Nếu không, cần denormalize trước
        new_labels.append((cls, coords))

    return img_resized, new_labels

# ---------------------------
# 5. Rotate + update label (FIX: normalized coordinates)
# ---------------------------
def rotate_90(img, labels):
    """Rotate 90 degrees clockwise"""
    img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Rotate 90: (x, y) -> (1-y, x)
            nx = 1 - y
            ny = x
            new_c += [nx, ny]
        new.append((cls, new_c))
    return img_r, new

def rotate_180(img, labels):
    """Rotate 180 degrees"""
    img_r = cv2.rotate(img, cv2.ROTATE_180)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Rotate 180: (x, y) -> (1-x, 1-y)
            new_c += [1-x, 1-y]
        new.append((cls, new_c))
    return img_r, new

def rotate_270(img, labels):
    """Rotate 270 degrees clockwise (90 counterclockwise)"""
    img_r = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Rotate 270: (x, y) -> (y, 1-x)
            nx = y
            ny = 1 - x
            new_c += [nx, ny]
        new.append((cls, new_c))
    return img_r, new

ROTATE_FUNCS = {
    90: rotate_90,
    180: rotate_180,
    270: rotate_270,
    360: lambda img, lab: (img.copy(), [l for l in lab])  # No rotation
}

# ---------------------------
# 6. Noise augmentation
# ---------------------------
def add_noise(img):
    """Add Gaussian noise"""
    noise = np.random.randn(*img.shape) * 10
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------
# 7. Extra augmentation - Horizontal Flip
# ---------------------------
def flip_h(img, labels):
    """Flip horizontally"""
    img_flipped = cv2.flip(img, 1)
    new = []
    for cls, coords in labels:
        nc = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Flip horizontal: (x, y) -> (1-x, y)
            nc.append(1 - x)
            nc.append(y)
        new.append((cls, nc))
    return img_flipped, new

EXTRA_AUGS = [flip_h]

# ---------------------------
# 8. Tạo folder dataset
# ---------------------------
print("\nCreating dataset directories...")
for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    os.makedirs(dataset_dir / sub, exist_ok=True)

# ---------------------------
# 9. Duyệt dataset gốc và augmentation
# ---------------------------
print("\nProcessing images and augmentation...")
final_data = []  # List of (unique_name, img, labels, subset)

img_paths = sorted(list(raw_img_dir.glob("*.jpg")))
print(f"Found {len(img_paths)} images in {raw_img_dir}")

for idx, img_path in enumerate(img_paths):
    name = img_path.stem
    lab_path = raw_lab_dir / f"{name}.txt"
    
    if not lab_path.exists():
        print(f"  [SKIP] {name}: No label file")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERROR] {name}: Cannot read image")
        continue
    
    labels = load_label(lab_path)
    if len(labels) == 0:
        print(f"  [SKIP] {name}: No labels")
        continue

    # Resize
    img_r, lab_r = resize_img_and_label(img, labels, image_size)

    # Generate augmented versions
    augmented = []
    
    # 1) Original resized
    augmented.append((f"{name}_orig", img_r, lab_r))
    
    # 2) Rotation variants
    for degree in [90, 180, 270]:
        img_rot, lab_rot = ROTATE_FUNCS[degree](img_r, lab_r)
        augmented.append((f"{name}_rot{degree}", img_rot, lab_rot))
    
    # 3) Add noise to some variants
    for i, (n, im, lb) in enumerate(list(augmented)):
        if random.random() < 0.5:  # 50% chance add noise
            img_noise = add_noise(im)
            augmented.append((f"{n}_noise", img_noise, lb))
    
    # 4) Add flip to some variants
    for i, (n, im, lb) in enumerate(list(augmented)):
        if random.random() < 0.3:  # 30% chance flip
            img_flip, lab_flip = flip_h(im, lb)
            augmented.append((f"{n}_flip", img_flip, lab_flip))
    
    final_data.extend(augmented)
    
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(img_paths)} images...")

print(f"\nTotal samples after augmentation: {len(final_data)}")

# ---------------------------
# 10. Chia train/val/test
# ---------------------------
print("\nSplitting dataset...")

# Lấy tên file gốc để split (tránh leak data giữa các tập)
original_names = list(set([name.split('_')[0] for name, _, _ in final_data]))
random.shuffle(original_names)

# Split original names
train_size = int(len(original_names) * train_ratio)
val_size = int(len(original_names) * val_ratio)

train_names = set(original_names[:train_size])
val_names = set(original_names[train_size:train_size + val_size])
test_names = set(original_names[train_size + val_size:])

print(f"  Original images: {len(original_names)}")
print(f"  Train: {len(train_names)} ({len(train_names)/len(original_names)*100:.1f}%)")
print(f"  Val: {len(val_names)} ({len(val_names)/len(original_names)*100:.1f}%)")
print(f"  Test: {len(test_names)} ({len(test_names)/len(original_names)*100:.1f}%)")

# Assign subset to each sample
def get_subset(name):
    orig_name = name.split('_')[0]
    if orig_name in train_names:
        return "train"
    elif orig_name in val_names:
        return "val"
    else:
        return "test"

# ---------------------------
# 11. Lưu ảnh và labels
# ---------------------------
print("\nSaving images and labels...")
subset_counts = {"train": 0, "val": 0, "test": 0}

for name, img, labels in final_data:
    subset = get_subset(name)
    subset_counts[subset] += 1
    
    # Save image
    img_path = dataset_dir / f"images/{subset}/{name}.jpg"
    cv2.imwrite(str(img_path), img)
    
    # Save label
    lab_path = dataset_dir / f"labels/{subset}/{name}.txt"
    save_label(lab_path, labels)

print(f"\nFinal dataset distribution:")
print(f"  Train: {subset_counts['train']} samples")
print(f"  Val: {subset_counts['val']} samples")
print(f"  Test: {subset_counts['test']} samples")

# ---------------------------
# 12. Tạo data.yaml
# ---------------------------
print("\nCreating data.yaml...")

data_config = {
    "train": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/train"),
    "val": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/val"),
    "test": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/test"),
    "nc": len(classes),
    "names": classes
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_config, f, sort_keys=False)

print(f"\nDataset created successfully!")
print(f"  Location: {dataset_dir}")
print(f"  Config: {data_yaml_path}")
print(f"  Classes: {classes}")
