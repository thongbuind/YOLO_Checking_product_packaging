import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.CamThread import CamThread
from utils.visual import FPSCalculator, make_grid
from utils.CamInfo import SlotInfo, CamInfo
from utils.process_results_from_yolo import process_results_from_yolo

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config" / "config.json"
model_file = project_root / "model" / "best.pt"

model = YOLO(model_file).to('mps:0')
model.fuse()

with open(config_file, 'r') as f:
    config = json.load(f)
image_size = config["image_size"]
classes = config["classes"]
slot_expected_items = {i: config[f"slot_{i}"] for i in range(1, 11)}
urls = [config[f"url_{i}"] for i in range(1, 5)]
cam_configs = [
    ("cam_1", 1),
    ("cam_2", 1),
    ("cam_3", 1),
    ("cam_4", 1)
]

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

cam_threads = {
    name: CamThread(name, source, mode="webcam") # hoặc mode="rtsp"
    for name, source in cam_configs
}

camera_locks = {
    cam_id: threading.Lock() 
    for cam_id in cameras.keys()
}

executor = ThreadPoolExecutor(max_workers=4)

fps_calc = FPSCalculator()

import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

# ====================== THỐNG KÊ YOLO ======================
yolo_times = []        # Lưu tất cả thời gian inference
yolo_count = 0
yolo_total_time = 0.0
yolo_max_time = 0.0

# ====================== THỐNG KÊ XỬ LÝ KẾT QUẢ ======================
process_times = []     # Lưu tất cả thời gian xử lý
process_count = 0
process_total_time = 0.0
process_max_time = 0.0

# ====================== VÒNG LẶP CHÍNH ======================
while True:
    fps_calc.update()
    frames = {}
    fps_values = {}
    yolo_results = {}
        
    for name, cam in cam_threads.items():
        frame, fps = cam.read()
        if frame is not None:
            frames[name] = frame
            fps_values[name] = fps
        
    if len(frames) != 4:
        continue
        
    cam_names = list(frames.keys())
    frame_list = [frames[n] for n in cam_names]
    
    # Đo thời gian YOLO batch inference
    yolo_start = time.time()
    batch_results = model(frame_list, imgsz=image_size, device="mps:0")
    yolo_time = (time.time() - yolo_start) * 1000  # ms
    
    # Cập nhật thống kê YOLO
    yolo_count += 1
    yolo_total_time += yolo_time
    if yolo_time > yolo_max_time:
        yolo_max_time = yolo_time
    
    yolo_avg_time = yolo_total_time / yolo_count
    yolo_times.append(yolo_time)
    
    print(f"[YOLO] Batch inference: {yolo_time:.2f}ms | Avg: {yolo_avg_time:.2f}ms | Max: {yolo_max_time:.2f}ms")

    # ⏱️ Đo thời gian xử lý kết quả (for loop + process_results)
    process_start = time.time()
        
    frames = process_results_from_yolo(
        frames=frames,
        batch_results=batch_results,
        cameras=cameras,
        cam_names=cam_names,
        classes=classes,
        camera_locks=camera_locks,
        executor=executor
    )
    
    process_time = (time.time() - process_start) * 1000  # ms
    
    # Cập nhật thống kê xử lý
    process_count += 1
    process_total_time += process_time
    if process_time > process_max_time:
        process_max_time = process_time
    
    process_avg_time = process_total_time / process_count
    process_times.append(process_time)
    
    print(f"[PROCESS] Results processing: {process_time:.2f}ms | Avg: {process_avg_time:.2f}ms | Max: {process_max_time:.2f}ms")

    sys_fps = fps_calc.get_fps()
    grid = make_grid(frames, fps_values, sys_fps)
    cv2.imshow("4 Cameras - Assembly Check System", grid)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================== KẾT THÚC & THỐNG KÊ ======================
print("\n[INFO] Shutting down...")

# Tính min time chính xác từ danh sách đã lưu
yolo_min_time = min(yolo_times) if yolo_times else 0.0
yolo_avg_time = yolo_total_time / yolo_count if yolo_count > 0 else 0.0

process_min_time = min(process_times) if process_times else 0.0
process_avg_time = process_total_time / process_count if process_count > 0 else 0.0

print("\n" + "="*60)
print("YOLO BATCH INFERENCE STATISTICS")
print("="*60)
print(f"Total inferences: {yolo_count}")
print(f"Average time:     {yolo_avg_time:.2f} ms")
print(f"Max time:         {yolo_max_time:.2f} ms")
print(f"Min time:         {yolo_min_time:.2f} ms")
print(f"Total time:       {yolo_total_time / 1000:.2f} s")
print("="*60 + "\n")

print("="*60)
print("RESULTS PROCESSING STATISTICS")
print("="*60)
print(f"Total processes:  {process_count}")
print(f"Average time:     {process_avg_time:.2f} ms")
print(f"Max time:         {process_max_time:.2f} ms")
print(f"Min time:         {process_min_time:.2f} ms")
print(f"Total time:       {process_total_time / 1000:.2f} s")
print("="*60 + "\n")

# Dọn dẹp tài nguyên
for cam in cam_threads.values():
    cam.release()
executor.shutdown(wait=True)
cv2.destroyAllWindows()
print("[INFO] System stopped")

# ====================== VẼ ĐỒ THỊ ======================
if yolo_count > 10:  # Đảm bảo có đủ dữ liệu sau khi bỏ 10 giá trị đầu
    from datetime import datetime
    
    # Bỏ 10 giá trị đầu tiên
    yolo_times_plot = yolo_times[10:]
    process_times_plot = process_times[10:]
    
    # Tính lại các giá trị thống kê cho phần còn lại
    yolo_avg_plot = np.mean(yolo_times_plot)
    yolo_max_plot = np.max(yolo_times_plot)
    yolo_min_plot = np.min(yolo_times_plot)
    
    process_avg_plot = np.mean(process_times_plot)
    process_max_plot = np.max(process_times_plot)
    process_min_plot = np.min(process_times_plot)
    
    inference_indices = np.arange(11, yolo_count + 1)  # Bắt đầu từ 11
    
    # ===== ĐỒ THỊ 1: YOLO Inference =====
    plt.figure(figsize=(12, 6))
    plt.plot(inference_indices, yolo_times_plot, label='Thời gian xử lý (ms)', color='blue', marker='o', markersize=3, linewidth=1.5)

    plt.axhline(y=yolo_avg_plot, color='green', linestyle='--', linewidth=2,
                label=f'Trung bình: {yolo_avg_plot:.2f} ms')
    plt.axhline(y=yolo_max_plot, color='red', linestyle='--', linewidth=2,
                label=f'Lớn nhất: {yolo_max_plot:.2f} ms')
    plt.axhline(y=yolo_min_plot, color='orange', linestyle='--', linewidth=2,
                label=f'Nhỏ nhất: {yolo_min_plot:.2f} ms')

    plt.title('Đồ thị thời gian xử lý YOLO qua từng lần inference (bỏ 10 lần đầu)', fontsize=14)
    plt.xlabel('Lần inference', fontsize=12)
    plt.ylabel('Thời gian (ms)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu đồ thị YOLO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yolo_filename = f"yolo_inference_time_{timestamp}.png"
    plt.savefig(yolo_filename, dpi=300, bbox_inches='tight')
    print(f"[INFO] YOLO chart saved as: {yolo_filename}")
    
    # ===== ĐỒ THỊ 2: Results Processing =====
    plt.figure(figsize=(12, 6))
    plt.plot(inference_indices, process_times_plot, label='Thời gian xử lý (ms)', color='purple', marker='o', markersize=3, linewidth=1.5)

    plt.axhline(y=process_avg_plot, color='green', linestyle='--', linewidth=2,
                label=f'Trung bình: {process_avg_plot:.2f} ms')
    plt.axhline(y=process_max_plot, color='red', linestyle='--', linewidth=2,
                label=f'Lớn nhất: {process_max_plot:.2f} ms')
    plt.axhline(y=process_min_plot, color='orange', linestyle='--', linewidth=2,
                label=f'Nhỏ nhất: {process_min_plot:.2f} ms')

    plt.title('Đồ thị thời gian xử lý kết quả qua từng lần (bỏ 10 lần đầu)', fontsize=14)
    plt.xlabel('Lần xử lý', fontsize=12)
    plt.ylabel('Thời gian (ms)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu đồ thị Processing
    process_filename = f"results_processing_time_{timestamp}.png"
    plt.savefig(process_filename, dpi=300, bbox_inches='tight')
    print(f"[INFO] Processing chart saved as: {process_filename}")
    
    # Hiển thị cả 2 đồ thị
    plt.show()
    
else:
    print("[INFO] Không đủ dữ liệu inference để vẽ đồ thị (cần > 10 lần).")
