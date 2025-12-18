import numpy as np
import cv2
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from logic.slot_position import slot_position

LABEL_EXT = ".txt"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

def load_slot_boxes(label_path):
    boxes = []
    if not os.path.exists(label_path):
        print(f"⚠️ Không có label: {label_path}")
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue

        cls = int(parts[0])
        if cls != 0:
            continue

        coords = [float(x) for x in parts[1:]]
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        
        cx = sum(p[0] for p in points) / 4
        cy = sum(p[1] for p in points) / 4
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)

        boxes.append((0, cx, cy, w, h))

    return boxes

def draw_slots(img, boxes, slot_res_tuple):
    H, W = img.shape[:2]
    img_draw = img.copy()

    slot_res, vec_connect, p_mid = slot_res_tuple

    for slot_num, box_id in slot_res.items():
        for b in boxes:
            if b[0] == box_id:
                cx, cy = int(b[1] * W), int(b[2] * H)
                break

        cv2.circle(img_draw, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(img_draw, str(slot_num), (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

    slot_to_idx = {b[0]: i for i, b in enumerate(boxes)}

    line3_idx = [slot_to_idx[slot_res[i]] for i in [1, 2, 3]]
    line2_idx = [slot_to_idx[slot_res[i]] for i in [4, 5]]

    centers = np.array([[b[1] * W, b[2] * H] for b in boxes])
    line3_px = [tuple(centers[i].astype(int)) for i in line3_idx]
    line2_px = [tuple(centers[i].astype(int)) for i in line2_idx]

    for i in range(2):
        cv2.line(img_draw, line3_px[i], line3_px[i + 1], (255, 0, 0), 3)

    cv2.line(img_draw, line2_px[0], line2_px[1], (0, 255, 0), 3)

    p1 = (int(p_mid[0] * W), int(p_mid[1] * H))
    p2 = (int((p_mid + vec_connect)[0] * W), int((p_mid + vec_connect)[1] * H))
    cv2.arrowedLine(img_draw, p1, p2, (0, 0, 255), 3, tipLength=0.1)

    cv2.putText(img_draw,
                f"dx={vec_connect[0]:.3f}, dy={vec_connect[1]:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    dx, dy = vec_connect
    if dx > 0 and dy > 0: case = "TH1"
    elif dx > 0 and dy < 0: case = "TH2"
    elif dx < 0 and dy < 0: case = "TH3"
    else: case = "TH4"

    cv2.putText(img_draw, case, (10, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

    return img_draw

def process_dataset(images_dir, labels_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    imgs = [f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS]

    for img_name in imgs:
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + LABEL_EXT)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không đọc được ảnh: {img_path}")
            continue

        boxes = load_slot_boxes(lbl_path)

        if len(boxes) != 5:
            print(f"❌ {img_name}: số lượng obj class 0 = {len(boxes)} → KHÔNG hợp lệ")
            continue

        boxes = [(i + 1, b[1], b[2], b[3], b[4]) for i, b in enumerate(boxes)]

        try:
            res = slot_position(1, boxes)
            out_img = draw_slots(img, boxes, res)

            save_path = os.path.join(save_dir, f"out_{img_name}")
            cv2.imwrite(save_path, out_img)

            print(f"✔ Xử lý xong: {save_path}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {img_name}: {e}")


images_dir = "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/dataset/images/val"
labels_dir = "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/dataset/labels/val"
save_dir   = "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/slot_position/result"

process_dataset(images_dir, labels_dir, save_dir)
