"""
QUY TRÌNH ĐÓNG GÓI:
    - Cam 1: Cho 3 "mach_nho" vào slot_1, slot_2 và slot_3.                             # slot_will_be_checked có 3 slot
    - Cam 2: Cho "mach_lon" vào slot_4, cho usb_to_jtag vào slot_5                      # slot_will_be_checked có 2 slot
    - Cam 3: Cho "day_black" vào slot_6, "day_lgbt" vào slot_7, "day_white" vào slot_8  # slot_will_be_checked có 3 slot
    - Cam 4: Cho "pack_circut" vào slot_9, "day_gray" vào slot_10                       # slot_will_be_checked có 2 slot

QUY TRÌNH XỬ LÝ:
    - Ban đầu, trạng thái của cam là waiting, chạy mô hình yolo.
    - Khi 5 slot xuất hiện, chuyển trạng thái sang checking và 
      chạy hàm slot_position để xác định slot nào là slot 1,2,3,4,5.
    - Khi 1 item xuất hiện và va chạm với 1 slot, chạy hàm check_item_and_slot.
      (hàm này sẽ nhận vào items, số thứ tự slot, toạ độ item, toạ độ slot, config. Nếu item 80% item nằm trong slot thì xác định item đã ở trong slot, 
      check config xem chúng có thuộc về nhau không, và trả về "true" hoặc "false").
    - Nếu "true", chuyển slot.state sang "oke", nếu "false" thì "wrong".
    - Khi tất cả các slot_will_be_checked trong cam có state "oke" thì cam.state được chuyển sang "done", 
      ngược lại nếu có ít nhất một slot.state là "wrong" thì chuyển cam.state sang "false".

CLASS SlotInfo:
    - Gồm id, expected(string), points (mảng np có 4 điểm xác định phạm vi slot), state(string)
    - Hàm set_state(), set_points()
    - Hàm get_state(), get_points()
    - Hàm update()
    
CLASS CamInfo:
    - Gồm cam_id, slot_will_be_checked, state, slots
    - Hàm init(id) dùng để khởi tạo thông tin cam (id=id, slot_will_be_checked lấy trong config, state="waiting", slots rỗng) trong file detect.py
    - Hàm set_state(new_state) dùng để thay đổi state của cam
    - Hàm add_slot()
    - Hàm get_slot()

WORK FLOW FILE detect.py:
    - 1. Mở 4 camera bằng Camthread
    - 2. Khởi tạo riêng biệt thông tin của mỗi camera bằng CamInfo.init()
    - 3. Khởi tạo YOLOProcesser
    - 4. Khởi tạo detection_results để lưu kết quả cuối cùng nhận được từ Yolo
    - 5. Chạy vòng lặp chính

MAIN LOOP TRONG FILE detect.py:
    - Đọc frames từ 4 camera đã mở từ bước 1
    - Gom 4 frames thành 1 batch, gửi cho YOLOProcesser xử lý
    - Lấy kết quả từ YOLOProcesser (object slot là 0, 1->8 là 8 items)
    - Nếu detection_results không trống, gọi hàm process_yolo_results để xử lý frame và
    - Tạo grid 2x2 và hiển thị frames

HÀM process_results_from_yolo:
    - Nhận đầu vào là frames (4 frame), detection_results và thông tin của 4 camera ở bước 2 bên trên
    - Tạo thread xử lý song song 4 camera cùng lúc (hàm process_single_camera cho mỗi cam)
    - Hàm process_single_camera:
        - Đầu vào là cam_id, frame và detection_results của cam đó, caminfo
        - Đầu tiên, kiểm tra detection_results:
            - Nếu không có đúng 5 slot xuất hiện, bỏ qua không xử lý gì cả
            - Nếu có 5 đúng slot, chạy hàm slot_position(), cập nhật info cho cả 5 slot
            - Dùng hàm set_state("checking")
        - Nếu trong detection_results có một item nào đó (1->8) thì chạy hàm is_item_in_slot_and_valid(), hàm này sẽ trả về true/false
        - Nếu "true" thì dùng set_state("oke"), "false" thì set_state("wrong")
        - If tất cả slot trong slot_will_be_checked có state "oke" (dùng get_state()) thì chuyển state của cam sang "done" (get_state("done")). Else get_state("false)
    - Vẽ lên màn hình các box mà detection_results có, hiển thị cam.state và các cam.slots.state (chỉ các slot có trong danh sách slot_will_be_checked)
"""

import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.CamThread import CamThread
from utils.YOLOProcesser import YOLOProcessor
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

cam_threads = {
    name: CamThread(name, source, mode="webcam") # hoặc mode="rtsp"
    for name, source in cam_configs
}
yolo_processor = YOLOProcessor(model, image_size, conf_threshold=0.7, max_fps=30)

slot_will_be_checked_of_cam_1 = [1, 2, 3]
slot_will_be_checked_of_cam_2 = [4, 5]
slot_will_be_checked_of_cam_3 = [6, 7, 8]
slot_will_be_checked_of_cam_4 = [9, 10]

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

camera_locks = {
    cam_id: threading.Lock() 
    for cam_id in cameras.keys()
}

executor = ThreadPoolExecutor(max_workers=4)

detection_results = {}
fps_calc = FPSCalculator()

try:
    while True:
        fps_calc.update()
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
            frames = process_results_from_yolo(
                frames=frames,
                yolo_results=detection_results,
                cameras=cameras,
                camera_locks=camera_locks,
                executor=executor
            )
        
        sys_fps = fps_calc.get_fps()
        grid = make_grid(frames, fps_values, detection_results, sys_fps)
        cv2.imshow("4 Cameras - Assembly Check System", grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\n[INFO] Shutting down...")
    for cam in cam_threads.values():
        cam.release()
    yolo_processor.stop()
    executor.shutdown(wait=True)
    cv2.destroyAllWindows()
    print("[INFO] System stopped")
