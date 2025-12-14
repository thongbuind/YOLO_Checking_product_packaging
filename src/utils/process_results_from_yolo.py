import numpy as np
import threading
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor

from utils.visual import draw_visualization
from utils.slot_position import slot_position
from utils.CamInfo import CamInfo

def bbox_area(points: np.ndarray) -> float:
    """Diện tích bbox"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def bbox_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Diện tích giao giữa 2 bbox"""
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        intersection = p1.intersection(p2)
        return intersection.area
    except:
        return 0.0

def is_item_in_slot(item_points: np.ndarray, slot_points: np.ndarray) -> bool:
    item_area = bbox_area(item_points)
    slot_area = bbox_area(slot_points)
    intersection_area = bbox_intersection_area(item_points, slot_points)
    overlap_item_ratio = intersection_area / item_area
    overlap_slot_ratio = intersection_area / slot_area

    if overlap_item_ratio >= 0.8:
        return True

    if overlap_slot_ratio >= 0.8:
        return True

    return False

def is_valid_item(item_name: str, expected_item: str) -> bool:
    return item_name == expected_item

def process_single_camera(yolo_results: dict, frame: np.ndarray, cam_info: CamInfo, cam_lock: threading.Lock):
    slots_boxes = yolo_results['slots_boxes']
    items_boxes = yolo_results['items_boxes']

    with cam_lock:
        if len(slots_boxes) == 0: 
        # khi chạy thật bằng camera thì để if len(slots_boxes) == 0 cũng được, để cover cho những frame yolo detect thiếu, thì vẫn còn data cũ
            cam_info.set_state("waiting")
            for slot_id, slot in cam_info.slots_list.items():
                slot.set_state("empty")
                slot.set_points(None)

        # nếu số slot = 5, chạy slot_position để gán id cho slot nớ, chuyển state của cam sang checking
        if len(slots_boxes) == 5:
            if cam_info.get_state() == "waiting":
                cam_info.set_state("checking")

            # xác định vị trí kèm id của các slot
            slot_mapping = slot_position(slots_boxes)
            cam_info.update_slot(slot_mapping, slots_boxes)
        
            # lặp qua từng item xuất hiện, xem hấn có trong slot mô ko, nếu có thì xem có valid ko, ghi trạng thái slot
            # độ phức tạp O(n^2) nhưng mà chắc là nỏ can chi vì số lượng object nhỏ
            for item_name, item_points in items_boxes:
                for slot_id in cam_info.slot_will_be_checked:
                    slot = cam_info.get_slot(slot_id)
                    slot_points = slot.get_points()                        
                    expected_item = slot.expected_item

                    if slot_points is not None and is_item_in_slot(item_points, slot_points):
                        if is_valid_item(item_name, expected_item):
                            slot.set_state("oke")
                        else:
                            slot.set_state("wrong")

            # kiểm tra theo slot 
            for slot_id in cam_info.slot_will_be_checked:
                slot = cam_info.get_slot(slot_id)
                slot_points = slot.get_points()
                found_item_in_slot = False

                for item_name, item_points in items_boxes:
                    if slot_points is not None and is_item_in_slot(item_points, slot_points):
                        found_item_in_slot = True
                        if is_valid_item(item_name, slot.expected_item):
                            slot.set_state("oke")
                        else:
                            slot.set_state("wrong")
                        #break # early stop, nhưng mà nỏ cần cụng đc, duyệt có tí mà

                if not found_item_in_slot:
                    slot.set_state("empty")

            # kiểm tra danh sách expected slot để chỉnh cam.state
            all_slot_is_oke = all(
                cam_info.get_slot(expected_slot_id).get_state() == "oke"
                for expected_slot_id in cam_info.slot_will_be_checked
                if cam_info.get_slot(expected_slot_id) is not None
            )

            had_one_slot_wrong = any(
                cam_info.get_slot(sid).get_state() == "wrong"
                for sid in cam_info.slot_will_be_checked
                if cam_info.get_slot(sid) is not None
           )

            had_one_slot_empty = any(
                cam_info.get_slot(sid).get_state() == "empty"
                for sid in cam_info.slot_will_be_checked
                if cam_info.get_slot(sid) is not None
            )
            
            if all_slot_is_oke: # tất cả đều oke
                cam_info.set_state("done")
            elif had_one_slot_wrong: # chỉ cần có 1 slot wrong
                cam_info.set_state("false")
            elif had_one_slot_empty: # nếu có 1 slot empty (và ko slot nào wrong đã lọc ở trên)
                cam_info.set_state("checking")
                
    draw_visualization(frame, cam_info, items_boxes)

def process_results_from_yolo(frames: dict, batch_results: dict, cameras: dict, cam_names, classes, camera_locks: dict, executor: ThreadPoolExecutor):
    futures = []
    yolo_results = {}

    for idx, cam_id in enumerate(cam_names):
        res = batch_results[idx]
        slots_boxes, items_boxes = [], []

        if res.obb is not None:
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
            cls_ids = res.obb.cls.cpu().numpy().astype(int)
            confs = res.obb.conf.cpu().numpy()

            for pts, cls_id, score in zip(xyxyxyxy, cls_ids, confs):
                pts = pts.astype(np.float32)

                if cls_id == 0 and score < 0.3:   
                    continue
                elif cls_id >= 1 and score < 0.7: 
                    continue

                if cls_id == 0:
                    slots_boxes.append(pts)
                else:
                    item_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                    items_boxes.append((item_name, pts))

        yolo_results[cam_id] = {
            "slots_boxes": slots_boxes,
            "items_boxes": items_boxes
        }

    for cam_name, detect_result in yolo_results.items():
        frame = frames.get(cam_name)
        cam_info = cameras.get(cam_name)
        cam_lock = camera_locks.get(cam_name)
        if frame is None or cam_info is None or cam_lock is None:
            continue
        futures.append(
            executor.submit(process_single_camera, detect_result, frame, cam_info, cam_lock)
        )
    
    for future in futures:
        future.result()

    return frames
