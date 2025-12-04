import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, field

from utils.slot_position import slot_position
from utils.visual import FPSCalculator, make_grid
from utils.CamThread import CamThread
from utils.YOLOProcesser import YOLOProcessor
# from utils.Cam_progress import Cam_1_progress, Cam_2_progress, Cam_3_progress, Cam_4_progress

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config.json"
model_file = project_root / "model" / "best.pt"

model = YOLO(model_file)
# model.fuse()

with open(config_file, 'r') as f:
    config = json.load(f)
image_size = config["image_size"]
urls = [config[f"url_{i}"] for i in range(1, 5)]

classes = config["classes"]
# tôi mới chuyển config từ "items": ["mach_nho", "mach_lon", "usb_to_jtag", "day_black", "day_lgbt", "day_white", "pack_circut", "day_gray"] sang "classes": ["slot", "mach_nho", "mach_lon", "usb_to_jtag", "day_black", "day_lgbt", "day_white", "pack_circut", "day_gray"] hãy chỉnh sửa những đoạn liên quan bên dứoi để code không bị ảnh hưởng, chú ý giữ nguyên toàn bộ logic

# Slot 1–10 → item name
slot_expected_items = {
    i: config[f"slot_{i}"]
    for i in range(1, 11)
}

cam_configs = [
    ("cam1", 0),
    ("cam2", 0),
    ("cam3", 0),  # giả lập cam3 = cam0
    ("cam4", 0)   # giả lập cam4 = cam1
]

cam_threads = {name: CamThread(name, url) for name, url in cam_configs}
yolo_processor = YOLOProcessor(model, image_size, conf_threshold=0.7)

slots = []
detection_results = {}

"""
Class camera gồm state (waiting, checking, done, false), 5 slot(empty, oke, wrong), 2-3 items (tuỳ cam) có tên lấy từ config trên, các hàm thao tác với các biến, để trống đấy tôi sẽ viết sau bao gồm: khi phát hiện có 5 vật thể xuất hiện trong khung hình thì chuyển state từ waiting sang checking, khi thấy 1 trong 8 items xuất hiện thì chạy hàm check(cứ tạo viết bừa tên hàm đi để đấy tôi viết sau), khi tất cả các slot đều ok thì chuyển state về done, nếu có 1 slot wrong thì chuyển state về false, khi không phát hiện được các slot thì chuyển về waiting
boxes là đầu ra của mô hình yolov8_oob gồm loại object, toạ độ gốc, kích thước và góc xoay của box
"""

@dataclass
class CameraState:
    state: str = "waiting"   # waiting -> checking -> done/false
    slots: list = field(default_factory=lambda: ["empty"] * 5)
    items: list = field(default_factory=list)

    # nếu khung hình xuất hiện 5 slot, chạy hàm này
    def on_detect_slots(self, boxes):
        # thêm hàm if để tránh lỗi, nếu không đủ 5 slot -> pass hàm
        slot_map, _, _ = slot_position(boxes)

        self.slots = slot_map

        if self.state == "waiting":
            self.state = "checking"

    # khi khung hình xuất hiện ít nhất 1 item thì gọi hàm này để theo dõi
    def on_detect_item(self, boxes):
        """
        boxes: danh sách các bounding box detect được từ YOLO, mỗi box chứa:
            - box_name: tên item detect (ví dụ "mach_nho")
            - xyxy: tọa độ [x1, y1, x2, y2]
    
        Logic:
            - Chỉ gọi khi phát hiện item (không cần đủ 5 slot)
            - Lấy các item detect được
            - Gọi follow_items(items_detected, slots)
        """
        if not boxes or len(boxes) == 0:
            return
    
        items_detected = []
        items_only = classes[1:]
        for box_name, xyxy in boxes:
            if box_name in items_only:
                items_detected.append({
                "name": box_name,
                "xyxy": xyxy
            })
    
        if items_detected:
            # Gọi hàm follow_items để xử lý va chạm slot và update state
            # self.slots = list các slot của cam, mỗi slot gồm:
            #   {"slot_id": int, "expected_item": str, "state": "empty/oke/false", "xyxy": [...]}
            self.follow_items(items_detected, self.slots)
    
    def update_state_done(self):
        """Khi tất cả slot đều oke → chuyển state = done."""
        pass

    def update_state_false(self):
        """Khi có 1 slot wrong → chuyển state = false."""
        pass

    def reset_to_waiting(self):
        """Khi không detect được đủ 5 slot → quay về waiting."""
        pass

display_fps = FPSCalculator()

"""
Đầu tiên, state mặc định của cả 4 cam là "Waiting" (sau này nếu có thể thì cố gắng giảm mức tiêu thụ tài nguyên của những cam này, nhưng hiện tại cứ để bình thường thế chưa cần quan tâm)
Cho mô hình YOLO chạy đồng thời ở cả 4 cam
Khi cam detect được 5 slot, chuyển sang state "Checking"
Thực hiện Cam_progress ... (4 cam có 4 hàm cam_n_progress riêng, cứ để hàm giả sử đấy viết sau)
Khi không còn object nào trong khung hình (trong 1s), gửi state Done_step_x hoặc False_step_x và chuyển cam_state sang Waiting
"""

try:
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
        
        # 2. Submit batch inference (non-blocking)
        cam_names = list(frames.keys())
        frame_list = [frames[name] for name in cam_names]
        yolo_processor.submit(cam_names, frame_list)
        
        # 3. Lấy kết quả detection nếu có (non-blocking)
        results = yolo_processor.get_results()
        if results is not None:
            detection_results = results
        # 4. Xử lý detection results
        for cam_name, det_data in detection_results.items():
            slot_boxes = det_data['slot_boxes']
            item_boxes = det_data['item_boxes']
            
            # Xử lý slot
            if len(slot_boxes) == 5:
                try:
                    slot_result, vec_connect, p_mid = slot_position(slot_boxes)
                    slots = slot_result
                except Exception as e:
                    print(f"Slot position error: {e}")
            
            # Xử lý item
            for item_name, item_xyxy in item_boxes:
                for slot_id, slot_xyxy in slots:
                    # ok = check_iou_item(item_xyxy, slot_xyxy)
                    pass
            
            # 5. Visualize trên frame (optional - có thể tách thread riêng)
            if cam_name in frames:
                frame = frames[cam_name]
                
                # Draw slot boxes
                for xyxy in slot_boxes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw item boxes
                for item_name, xyxy in item_boxes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, item_name, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 6. Overlay FPS info
        for name, frame in frames.items():
            fps = fps_values.get(name, 0)
            cv2.putText(frame, f"{name} | {fps:.1f} FPS", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        sys_fps = display_fps.get_fps()
        grid = make_grid(frames, fps_values, detection_results, slots, sys_fps)

        # Overall system FPS
        sys_fps = display_fps.get_fps()
        cv2.putText(grid, f"System FPS: {sys_fps:.1f}", (10, grid.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Grid 2x2 - 4 Cameras", grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    yolo_processor.stop()
    for cam in cam_threads.values():
        cam.release()
    cv2.destroyAllWindows()
    print("System stopped.")
