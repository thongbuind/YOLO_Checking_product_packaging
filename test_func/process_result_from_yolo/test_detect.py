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

model = YOLO(MODEL_FILE).to("mps:0")
model.fuse()

slot_will_be_checked_of_cam_1 = [1, 2, 3]
slot_will_be_checked_of_cam_2 = [1, 2, 3, 4, 5]
slot_will_be_checked_of_cam_3 = [6, 7, 8]
slot_will_be_checked_of_cam_4 = [6, 7, 8, 9, 10]

slots = {i: SlotInfo(expected_item=slot_expected_items[i]) for i in range(1, 11)}

slots_list_for_cam_12 = {i: slots[i] for i in range(1, 6)}
slots_list_for_cam_34 = {i: slots[i] for i in range(6, 11)}

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

# ================== MAIN ==================
print("\n=== START TEST (cam_12 / cam_34) ===\n")

total_time = 0
total_frames = 0

for idx in range(max_len):
    frames = {}

    # -------- LOAD FRAMES --------
    for cam_id, imgs in cam_images.items():
        if idx < len(imgs):
            frame = cv2.imread(str(imgs[idx]))
            if frame is None:
                frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frames[cam_id] = frame

    # -------- YOLO BATCH INFERENCE (giống detect.py) --------
    cam_names = list(frames.keys())
    frame_list = [frames[n] for n in cam_names]
    
    batch_results = model(frame_list, imgsz=IMAGE_SIZE, device="mps:0")
    
    # -------- XỬ LÝ KẾT QUẢ (giống detect.py) --------
    start = time.time()
    
    frames = process_results_from_yolo(
        frames=frames,
        batch_results=batch_results,
        cameras=cameras,
        cam_names=cam_names,
        classes=CLASSES,
        camera_locks=camera_locks,
        executor=executor
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
