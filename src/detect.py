import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from utils.CamThread import CamThread
from utils.YOLOProcesser import YOLOProcessor
from utils.CameraInfo import CameraInfo
from utils.visual import FPSCalculator, make_grid
from utils.debug import setup_logger
from utils.process_yolo_result import process_yolo_results

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config" / "config.json"
model_file = project_root / "model" / "best.pt"

model = YOLO(model_file).to('mps')
model.fuse()

with open(config_file, 'r') as f:
    config = json.load(f)
image_size = config["image_size"]
classes = config["classes"]
slot_expected_items = {
    i: config[f"slot_{i}"] for i in range(1, 11)
}
urls = [config[f"url_{i}"] for i in range(1, 5)]
cam_configs = [
    ("cam_1", 1),
    ("cam_2", 1),
    ("cam_3", 1),
    ("cam_4", 1)
]
# cam_configs = [
#     ("cam_1", urls[1]),
#     ("cam_2", urls[2]),
#     ("cam_3", urls[3]),
#     ("cam_4", urls[4])
# ]

cam_threads = {
    name: CamThread(name, source, mode="webcam")
    for name, source in cam_configs
}
yolo_processor = YOLOProcessor(model, image_size, conf_threshold=0.7, max_fps=30)

camera_states = {
    "cam_1": CameraInfo(cam_id=1, expected_slots=slot_expected_items),
    "cam_2": CameraInfo(cam_id=2, expected_slots=slot_expected_items),
    "cam_3": CameraInfo(cam_id=3, expected_slots=slot_expected_items),
    "cam_4": CameraInfo(cam_id=4, expected_slots=slot_expected_items),
}

display_fps = FPSCalculator()

# debug
logger = setup_logger('/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/main_loop/debug_log.txt')
logger.info("=== ASSEMBLY CHECK SYSTEM STARTED ===")
logger.info(f"Using YOLO model: {model_file}")
logger.info(f"Cameras: {list(cam_threads.keys())}")

"""
KHỞI TẠO:
    - cam_1 với id=1, state="waiting", list_slot gồm 5 slot từ 1 đến 5
    - cam_2 với id=2, state="waiting", list_slot gồm 5 slot từ 1 đến 5
    - cam_3 với id=3, state="waiting", list_slot gồm 5 slot từ 6 đến 10
    - cam_4 với id=4, state="waiting", list_slot gồm 5 slot từ 6 đến 10
"""

"""
MAIN LOOP:
    - Đọc frame từ 4 camera.
    - Gom các frame thành 1 batch, gửi cho YOLO xử lý.
    - Lấy kết quả từ YOLO.
    - Xử lý kết quả.
    - Tạo grid và hiển thị
"""

# MAIN LOOP
detection_results = {}
while True:
    display_fps.update()
    frames = {} 
    fps_values = {}
        
    for name, cam in cam_threads.items():
        frame, fps = cam.read()
        if frame is not None:
            frames[name] = frame
            fps_values[name] = fps
    
    if len(frames) != 4:
        continue
    
    cam_names = list(frames.keys())
    frame_list = [frames[n] for n in cam_names]
    yolo_processor.submit(cam_names, frame_list)
    
    new_results = yolo_processor.get_results()
    if new_results is not None:
        detection_results = new_results

    if detection_results:
        frames = process_yolo_results(
            frames=frames,
            yolo_results=detection_results,
            camera_states=camera_states
        )

    sys_fps = display_fps.get_fps()
    grid = make_grid(frames, fps_values, detection_results, sys_fps)
    cv2.imshow("4 Cameras - Assembly Check System", grid)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
