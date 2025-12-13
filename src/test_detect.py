import cv2
import json
import time
import numpy as np
import threading
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from utils.process_results_from_yolo import process_results_from_yolo
from utils.CamInfo import SlotInfo, CamInfo

# ================== PATH ==================
MODEL_FILE = Path("/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/model/best.pt")
CONFIG_FILE = Path("/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/config/config.json")

data_for_cam_1 = Path("/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/process_single_camera/data/cam_12")
data_for_cam_2 = data_for_cam_1
data_for_cam_3 = Path("/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/process_single_camera/data/cam_34")
data_for_cam_4 = data_for_cam_3

DATA_DIRS = {
    "cam_1": data_for_cam_1,
    "cam_2": data_for_cam_2,
    "cam_3": data_for_cam_3,
    "cam_4": data_for_cam_4,
}

output_root = Path("/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/process_single_camera/output")
output_root.mkdir(exist_ok=True)

OUTPUT_DIRS = {
    "cam_1": output_root / "cam_1",
    "cam_2": output_root / "cam_2",
    "cam_3": output_root / "cam_3",
    "cam_4": output_root / "cam_4",
}
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ================== LOAD CONFIG ==================
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

CLASSES = config["classes"]
IMAGE_SIZE = 640
VALID_EXT = [".jpg", ".png", ".jpeg"]

slot_expected_items = {i: config[f"slot_{i}"] for i in range(1, 11)}

# ================== MODEL ==================
model = YOLO(MODEL_FILE).to("mps:0")
model.fuse()

slot_will_be_checked_of_cam_1 = [1, 2, 3]
slot_will_be_checked_of_cam_2 = [1, 2, 3, 4, 5]
slot_will_be_checked_of_cam_3 = [6, 7, 8]
slot_will_be_checked_of_cam_4 = [6, 7, 8, 9, 10]

slots = {
    1: SlotInfo(expected_item=slot_expected_items[1]),
    2: SlotInfo(expected_item=slot_expected_items[2]),
    3: SlotInfo(expected_item=slot_expected_items[3]),
    4: SlotInfo(expected_item=slot_expected_items[4]),
    5: SlotInfo(expected_item=slot_expected_items[5]),
    6: SlotInfo(expected_item=slot_expected_items[6]),
    7: SlotInfo(expected_item=slot_expected_items[7]),
    8: SlotInfo(expected_item=slot_expected_items[8]),
    9: SlotInfo(expected_item=slot_expected_items[9]),
    10: SlotInfo(expected_item=slot_expected_items[10]),
}

slots_list_for_cam_12 = {
    1: slots[1],
    2: slots[2],
    3: slots[3],
    4: slots[4],
    5: slots[5]
}
slots_list_for_cam_34 = {
    6: slots[6],
    7: slots[7],
    8: slots[8],
    9: slots[9],
    10: slots[10]
}
cameras = {
    "cam_1": CamInfo(slot_will_be_checked=slot_will_be_checked_of_cam_1, slots_list=slots_list_for_cam_12),
    "cam_2": CamInfo(slot_will_be_checked=slot_will_be_checked_of_cam_2, slots_list=slots_list_for_cam_12),
    "cam_3": CamInfo(slot_will_be_checked=slot_will_be_checked_of_cam_3, slots_list=slots_list_for_cam_34),
    "cam_4": CamInfo(slot_will_be_checked=slot_will_be_checked_of_cam_4, slots_list=slots_list_for_cam_34),
}

camera_locks = {cam: threading.Lock() for cam in cameras}
executor = ThreadPoolExecutor(max_workers=4)

# ================== LOAD IMAGE LIST ==================
def list_images(folder):
    if not folder.exists():
        return []
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in VALID_EXT])

cam_images = {
    cam_id: list_images(path)
    for cam_id, path in DATA_DIRS.items()
}

max_len = max(len(v) for v in cam_images.values())

# ================== UTILS ==================
def black_frame():
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

# ================== MAIN ==================
print("\n=== START TEST (cam_12 / cam_34) ===\n")

total_time = 0
total_frames = 0

for idx in range(max_len):
    frames = {}
    yolo_results = {}

    # -------- LOAD FRAME --------
    for cam_id, imgs in cam_images.items():
        if idx < len(imgs):
            frame = cv2.imread(str(imgs[idx]))
            if frame is None:
                frame = black_frame()
        else:
            frame = black_frame()

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frames[cam_id] = frame

    CONF_BY_CLASS = {
        0: 0.4,
    }
    DEFAULT_CONF = 0.7
    # -------- YOLO --------
    for cam_id, frame in frames.items():
        res = model(frame, imgsz=IMAGE_SIZE, conf=0.3, device="mps")[0]

        slots_boxes, items_boxes = [], []

        if res.obb is not None:
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
            cls_ids  = res.obb.cls.cpu().numpy().astype(int)
            confs    = res.obb.conf.cpu().numpy()

            for pts, cls_id, score in zip(xyxyxyxy, cls_ids, confs):
                min_conf = CONF_BY_CLASS.get(cls_id, DEFAULT_CONF)
                if score < min_conf:
                    continue

                pts = pts.astype(np.float32)

                if cls_id == 0:
                    slots_boxes.append(pts)
                else:
                    item_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
                    items_boxes.append((item_name, pts))

        yolo_results[cam_id] = {
            "slots_boxes": slots_boxes,
            "items_boxes": items_boxes
        }

    # -------- LOGIC --------
    start = time.time()
    frames = process_results_from_yolo(
        frames, yolo_results, cameras, camera_locks, executor
    )
    elapsed = (time.time() - start) * 1000

    total_time += elapsed
    total_frames += 1

    print(f"[FRAME {idx}] {elapsed:.2f} ms")

    # -------- SAVE --------
    for cam_id, frame in frames.items():
        cv2.imwrite(
            str(OUTPUT_DIRS[cam_id] / f"frame_{idx:04d}.jpg"),
            frame
        )

# ================== STATS ==================
print("\n===== DONE =====")
print(f"Frames: {total_frames}")
print(f"Avg: {total_time/total_frames:.2f} ms")
print(f"FPS(sim): {1000/(total_time/total_frames):.2f}")

executor.shutdown(wait=True)
