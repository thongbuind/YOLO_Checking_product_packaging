# import cv2
# import numpy as np
# import time
# from utils.slot_position import slot_position

"""
QUY TRÌNH CHUNG:
    - Ban đầu, hiển thị trạng thái là waiting, chạy mô hình yolo.
    - Khi 1 slot xuất hiện, chuyển trạng thái sang checking, 
      nếu đủ 5 slot thì chạy hàm slot_position để xác định slot nào là slot 1,2,3,4,5.
    - Khi 1 item xuất hiện và va chạm với 1 slot, chạy hàm check_item_and_slot.
      (hàm này sẽ nhận vào items, số thứ tự slot, toạ độ item, toạ độ slot, config. Nếu item 80% item nằm trong slot thì xác định item đã ở trong slot, 
      check config xem chúng có thuộc về nhau không, và trả về "valid" hoặc "invalid").
    - Nếu "valid", chuyển slot.state sang "oke", nếu "invalid" thì "wrong".
    - Khi tất cả các slot trong cam có state "oke" thì cam.state được chuyển sang "done", 
      ngược lại nếu có ít nhất một slot.state là "wrong" thì chuyển cam.state sang "false".
"""

"""
CAM 1: Nhiệm vụ là cho 3 "mach_nho" vào slot_1, slot_2 và slot_3
CAM 2: Cho "mach_lon" vào slot_4, cho usb_to_jtag vào slot_5
CAM 3: Cho "day_black" vào slot_6, "day_lgbt" vào slot_7, "day_white" vào slot_8
CAM 4: Cho "pack_circut" vào slot_9, "day_gray" vào slot_10
"""

# def process_yolo_results(yolo_results, frames, camera_states):
#     for cam_name, det_data in yolo_results.items():
#         if cam_name not in frames:
#             continue

#         frame = frames[cam_name]
#         H, W = frame.shape[:2]
#         slot_boxes = det_data['slot_boxes']
#         item_boxes = det_data['item_boxes']

#         state = camera_states[cam_name]
#         cam_id = state.cam_id

#         # --- Mapping slots lần đầu ---
#         if len(slot_boxes) == 5 and state.slot_mapping is None:
#             try:
#                 boxes_formatted = []
#                 for i, points in enumerate(slot_boxes):
#                     pts = np.array(points).reshape(4, 2)
#                     cx, cy = pts.mean(axis=0)
#                     xmin, ymin = pts.min(axis=0)
#                     xmax, ymax = pts.max(axis=0)
#                     w = xmax - xmin
#                     h = ymax - ymin
#                     boxes_formatted.append((i, cx, cy, w, h))

#                 mapping, _, _ = slot_position(cam_id, boxes_formatted)
#                 state.set_slot_mapping(mapping, slot_boxes)
#                 state.state = "checking"
#                 print(f"\n[INFO] {cam_name.upper()} → Phát hiện đủ 5 slot, chuyển sang CHECKING\n")

#             except Exception as e:
#                 print(f"[ERROR] slot_position {cam_name}: {e}")

#         # --- Update tọa độ slot khi box di chuyển ---
#         if state.slot_mapping and len(slot_boxes) == 5:
#             state.update_slot_coordinates(slot_boxes)

#         # --- Process items ---
#         if state.state == "checking" and state.slot_mapping is not None and item_boxes:
#             state.process_items(item_boxes)

#         # --- Kiểm tra khi box rời khỏi khung hình ---
#         if (state.state in ["checking", "done", "false"] and
#             state.slot_mapping is not None and
#             len(slot_boxes) == 0):

#             if state.pending_result is None:
#                 if state.is_all_required_ok():
#                     state.start_pending_result(success=True)
#                     state.state = "done"

#                 elif state.has_wrong_slot():
#                     wrong = state.get_wrong_slots()
#                     state.start_pending_result(success=False, wrong_slots=wrong)
#                     state.state = "false"
#                 else:
#                     pass

#         # --- Visualization ---
#         for points in slot_boxes:
#             points = np.int32(points)
#             cv2.polylines(frame, [points], True, (0, 255, 0), 3)
#             cx = int(np.mean(points[:, 0]))
#             cy = int(np.mean(points[:, 1]))
#             cv2.putText(frame, "Slot", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#         for item_name, points in item_boxes:
#             points = np.int32(points)
#             cv2.polylines(frame, [points], True, (255, 255, 0), 2)
#             cx = int(np.mean(points[:, 0]))
#             cy = int(np.mean(points[:, 1]))
#             cv2.putText(frame, item_name, (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

#         state_text = f"{cam_name}: {state.state.upper()}"
#         if state.pending_result:
#             elapsed = time.time() - state.pending_timestamp
#             state_text += f" (PENDING {elapsed:.1f}s)"

#         cv2.putText(frame, state_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#         if state.slot_mapping:
#             y_offset = 60
#             required = state.get_required_slots()
#             for slot_id in required:
#                 info = state.slot_mapping[slot_id]
#                 color = (0,255,0) if info.state == "oke" else (0,0,255) if info.state == "wrong" else (200,200,200)
#                 text = f"S{slot_id}: {info.state}"
#                 if info.placed_item:
#                     text += f" ({info.placed_item})"
#                 cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 y_offset += 25

#     return frames





import cv2
import numpy as np
import time
from utils.slot_position import slot_position
from utils.checking_items import calculate_iou_obb, has_collision, is_item_in_slot

"""
QUY TRÌNH CHUNG:
    - Ban đầu, hiển thị trạng thái là waiting, chạy mô hình yolo.
    - Khi 5 slot xuất hiện, chuyển trạng thái sang checking và 
      chạy hàm slot_position để xác định slot nào là slot 1,2,3,4,5.
    - Khi 1 item xuất hiện và va chạm với 1 slot, chạy hàm check_item_and_slot.
      (hàm này sẽ nhận vào items, số thứ tự slot, toạ độ item, toạ độ slot, config. Nếu item 80% item nằm trong slot thì xác định item đã ở trong slot, 
      check config xem chúng có thuộc về nhau không, và trả về "valid" hoặc "invalid").
    - Nếu "valid", chuyển slot.state sang "oke", nếu "invalid" thì "wrong".
    - Khi tất cả các slot trong cam có state "oke" thì cam.state được chuyển sang "done", 
      ngược lại nếu có ít nhất một slot.state là "wrong" thì chuyển cam.state sang "false".
"""

"""
CAM 1: Nhiệm vụ là cho 3 "mach_nho" vào slot_1, slot_2 và slot_3
CAM 2: Cho "mach_lon" vào slot_4, cho usb_to_jtag vào slot_5
CAM 3: Cho "day_black" vào slot_6, "day_lgbt" vào slot_7, "day_white" vào slot_8
CAM 4: Cho "pack_circut" vào slot_9, "day_gray" vào slot_10
"""

class CameraProcessor:
    """Class chứa toàn bộ business logic xử lý camera"""
    
    def __init__(self, delay_duration: float = 3.0):
        self.delay_duration = delay_duration

    def should_initialize_slots(self, camera_state, slot_boxes_count: int) -> bool:
        """Kiểm tra có nên khởi tạo slots không"""
        return slot_boxes_count == 5 and camera_state.slot_mapping is None

    def initialize_camera_slots(self, camera_state, slot_boxes: list):
        """Khởi tạo slots cho camera"""
        try:
            boxes_formatted = []
            for i, points in enumerate(slot_boxes):
                pts = np.array(points).reshape(4, 2)
                cx, cy = pts.mean(axis=0)
                xmin, ymin = pts.min(axis=0)
                xmax, ymax = pts.max(axis=0)
                w = xmax - xmin
                h = ymax - ymin
                boxes_formatted.append((i, cx, cy, w, h))

            mapping, _, _ = slot_position(camera_state.cam_id, boxes_formatted)
            camera_state.initialize_slots(mapping, slot_boxes)
            camera_state.change_state("checking")
            return True
        except Exception as e:
            print(f"[ERROR] Không thể khởi tạo slots: {e}")
            return False

    def update_slot_coordinates(self, camera_state, current_boxes: list):
        """Cập nhật tọa độ các slots dựa trên IoU"""
        if not camera_state.slot_mapping:
            return
        
        for slot_id, info in camera_state.slot_mapping.items():
            old_points = info.points
            best_iou = 0
            best_box = old_points
            
            for new_box in current_boxes:
                iou = calculate_iou_obb(old_points, new_box)
                if iou > best_iou:
                    best_iou = iou
                    best_box = new_box
            
            if best_iou > 0.5:
                camera_state.update_slot(slot_id, points=np.array(best_box, dtype=np.float32))

    def process_items(self, camera_state, item_boxes: list):
        """
        Xử lý items được detect, match với slots
        item_boxes: list of (item_name, points_array)
        """
        if not camera_state.slot_mapping:
            return
        
        for item_name, item_points in item_boxes:
            try:
                pts = np.array(item_points, dtype=np.float32).reshape(-1, 2)
                if pts.shape != (4, 2):
                    continue
                item_pts = pts
            except:
                continue
            
            best_slot_id = None
            best_score = 0
    
            for slot_id, info in camera_state.slot_mapping.items():
                # Skip slots đã xác định
                if info.state in ["oke", "wrong"]:
                    continue
            
                # Kiểm tra va chạm
                if not has_collision(item_pts, info.points):
                    continue
            
                # Tính điểm match
                score = is_item_in_slot(item_pts, info.points, threshold=0.8)
                if score > best_score:
                    best_score = score
                    best_slot_id = slot_id
            
            # Nếu tìm thấy slot phù hợp (>= 80%)
            if best_slot_id and best_score >= 0.8:
                slot = camera_state.get_slot(best_slot_id)
                
                # Kiểm tra item có đúng không
                if item_name == slot.expected:
                    camera_state.update_slot(best_slot_id, state="oke", placed_item=item_name)
                    print(f"  ✓ Cam {camera_state.cam_id} - Slot {best_slot_id}: {item_name} ĐÚNG (score={best_score:.2f})")
                else:
                    camera_state.update_slot(best_slot_id, state="wrong", placed_item=item_name)
                    print(f"  ✗ Cam {camera_state.cam_id} - Slot {best_slot_id}: {item_name} SAI (expected: {slot.expected}, score={best_score:.2f})")

    def check_all_required_ok(self, camera_state) -> bool:
        """Kiểm tra tất cả slot bắt buộc đã OK"""
        if not camera_state.slot_mapping:
            return False
        
        required = camera_state.get_required_slot_ids()
        return all(camera_state.get_slot(sid).state == "oke" for sid in required)

    def has_wrong_slot(self, camera_state) -> bool:
        """Kiểm tra có slot nào sai không"""
        if not camera_state.slot_mapping:
            return False
        
        required = camera_state.get_required_slot_ids()
        return any(camera_state.get_slot(sid).state == "wrong" for sid in required)

    def get_wrong_slots(self, camera_state) -> list:
        """Lấy danh sách slot sai"""
        if not camera_state.slot_mapping:
            return []
        
        required = camera_state.get_required_slot_ids()
        return [sid for sid in required if camera_state.get_slot(sid).state == "wrong"]

    def handle_box_disappeared(self, camera_state):
        """Xử lý khi box rời khỏi khung hình"""
        if camera_state.pending_result is not None:
            return  # Đã có pending result rồi
        
        if self.check_all_required_ok(camera_state):
            snapshot = camera_state.get_snapshot()
            camera_state.set_pending_result(
                success=True,
                snapshot=snapshot,
                timestamp=time.time()
            )
            camera_state.change_state("done")
            print(f"[PENDING] Cam {camera_state.cam_id} - Bắt đầu chờ {self.delay_duration}s trước khi gửi: DONE")
        
        elif self.has_wrong_slot(camera_state):
            wrong = self.get_wrong_slots(camera_state)
            snapshot = camera_state.get_snapshot()
            camera_state.set_pending_result(
                success=False,
                wrong_slots=wrong,
                snapshot=snapshot,
                timestamp=time.time()
            )
            camera_state.change_state("false")
            print(f"[PENDING] Cam {camera_state.cam_id} - Bắt đầu chờ {self.delay_duration}s trước khi gửi: FALSE")

    def check_pending_result(self, camera_state, slot_boxes_count: int):
        """
        Kiểm tra pending result:
        Returns:
            - None: chưa có gì
            - ("send", result_dict): cần gửi kết quả
            - ("cancel",): hủy gửi
        """
        if not camera_state.pending_result:
            return None
        
        elapsed = time.time() - camera_state.pending_timestamp
        
        # Case 1: Box quay lại trước khi hết thời gian
        if slot_boxes_count == 5:
            current_snapshot = camera_state.get_snapshot()
            old_snapshot = camera_state.pending_result["snapshot"]
   
            # State giống cũ → hủy gửi
            if current_snapshot == old_snapshot:
                print(f"[CANCEL] Cam {camera_state.cam_id} - Box quay lại với state cũ, hủy gửi")
                camera_state.clear_pending_result()
                return ("cancel",)
            
            # State khác → gửi kết quả cũ ngay
            else:
                print(f"[SEND NOW] Cam {camera_state.cam_id} - Box quay lại với state mới, gửi kết quả cũ")
                result_to_send = camera_state.pending_result.copy()
                camera_state.clear_pending_result()
                return ("send", result_to_send)
        
        # Case 2: Đủ thời gian → gửi kết quả
        elif elapsed >= self.delay_duration:
            print(f"[SEND] Cam {camera_state.cam_id} - Đủ {self.delay_duration}s, gửi kết quả")
            result_to_send = camera_state.pending_result.copy()
            camera_state.clear_pending_result()
            return ("send", result_to_send)
        
        return None

# Singleton processor instance
processor = CameraProcessor(delay_duration=3.0)

def process_yolo_results(yolo_results, frames, camera_states):
    """
    Hàm chính xử lý YOLO results cho tất cả cameras
    """
    for cam_name, det_data in yolo_results.items():
        if cam_name not in frames:
            continue

        frame = frames[cam_name]
        slot_boxes = det_data['slot_boxes']
        item_boxes = det_data['item_boxes']
        state = camera_states[cam_name]

        # --- 1. Khởi tạo slots lần đầu ---
        if processor.should_initialize_slots(state, len(slot_boxes)):
            success = processor.initialize_camera_slots(state, slot_boxes)
            if success:
                print(f"\n[INFO] {cam_name.upper()} → Phát hiện đủ 5 slot, chuyển sang CHECKING\n")

        # --- 2. Cập nhật tọa độ slots ---
        if state.slot_mapping and len(slot_boxes) == 5:
            processor.update_slot_coordinates(state, slot_boxes)

        # --- 3. Xử lý items ---
        if state.state == "checking" and state.slot_mapping and item_boxes:
            processor.process_items(state, item_boxes)

        # --- 4. Xử lý khi box biến mất ---
        if (state.state in ["checking", "done", "false"] and
            state.slot_mapping is not None and
            len(slot_boxes) == 0):
            processor.handle_box_disappeared(state)

        # --- 5. Kiểm tra pending result ---
        pending_action = processor.check_pending_result(state, len(slot_boxes))
        if pending_action:
            action_type = pending_action[0]
            if action_type == "send":
                result = pending_action[1]
                # TODO: Gửi kết quả đi (MQTT, API, etc.)
                print(f"[RESULT] Cam {state.cam_id}: {result}")
                state.reset()
            elif action_type == "cancel":
                # Pending bị hủy, tiếp tục checking
                pass

        # --- 6. Visualization ---
        draw_visualization(frame, state, slot_boxes, item_boxes, cam_name)

    return frames


def draw_visualization(frame, state, slot_boxes, item_boxes, cam_name):
    """Vẽ visualization lên frame"""
    # Vẽ slot boxes
    for points in slot_boxes:
        points = np.int32(points)
        cv2.polylines(frame, [points], True, (0, 255, 0), 3)
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))
        cv2.putText(frame, "Slot", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Vẽ item boxes
    for item_name, points in item_boxes:
        points = np.int32(points)
        cv2.polylines(frame, [points], True, (255, 255, 0), 2)
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))
        cv2.putText(frame, item_name, (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Vẽ state text
    state_text = f"{cam_name}: {state.state.upper()}"
    if state.pending_result:
        elapsed = time.time() - state.pending_timestamp
        state_text += f" (PENDING {elapsed:.1f}s)"

    cv2.putText(frame, state_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Vẽ thông tin slots
    if state.slot_mapping:
        y_offset = 60
        required = state.get_required_slot_ids()
        for slot_id in required:
            info = state.get_slot(slot_id)
            color = (0,255,0) if info.state == "oke" else (0,0,255) if info.state == "wrong" else (200,200,200)
            text = f"S{slot_id}: {info.state}"
            if info.placed_item:
                text += f" ({info.placed_item})"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
