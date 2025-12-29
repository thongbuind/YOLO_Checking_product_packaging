import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils.visual import draw_visualization
from process.slot_position import slot_position
from process.predict_slot import predict_slot
from camera.CamInfo import CamInfo
from utils.caculate import is_item_in_slot, is_valid_item

def process_single_camera(yolo_results: dict, frame: np.ndarray, cam_info: CamInfo):
    full_box_boxes = yolo_results['full_box_boxes']
    slots_boxes = yolo_results['slots_boxes']
    items_boxes = yolo_results['items_boxes']

    if len(slots_boxes) == 0: # thì reset state
    # if len(slots_boxes) == 0 hay if len(slots_boxes) != 5 đều được
    # nhưng khi chạy thực tế thì nên dùng == 0 hơn vì có thể cover cho những frame yolo detect thiếu, thì vẫn còn data cũ
        cam_info.set_state("waiting")
        for slot_id, slot in cam_info.slots_list.items():
            slot.set_state("empty")
            slot.set_points(None)

    else:
        if cam_info.get_state() == "waiting":
            cam_info.set_state("checking")

        # nếu ko đủ 5 slot thì đoán mấy slot còn thiếu
        if len(slots_boxes) < 5:
            layout_type = 0
            if min(cam_info.slot_will_be_checked) == 1:
                layout_type = 12
            elif min(cam_info.slot_will_be_checked) == 6:
                layout_type = 34
                
            cam_info.update_slot(predict_slot(layout_type, full_box_boxes, slots_boxes))

        # nếu số slot = 5, chạy slot_position để gán id cho slot nớ
        elif len(slots_boxes) == 5:
            cam_info.update_slot(slot_position(slots_boxes))

        # từ đoạn ni là chỉ lấy dữ liệu từ cam_info
        # lặp qua các slot, xem hn có chứa item mô ko, nếu có thì check xem có valid ko
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

            if not found_item_in_slot:
                slot.set_state("empty")

        # kiểm tra danh sách expected slot để set cam.state
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
            
        if all_slot_is_oke: # cấy mô cụng đều oke
            cam_info.set_state("done")
        elif had_one_slot_wrong: # chỉ cần có 1 slot wrong
            cam_info.set_state("false")
        elif had_one_slot_empty: # nếu có 1 slot empty (và nỏ slot mô wrong đạ lọc trên nớ)
            cam_info.set_state("checking")

    draw_visualization(frame, cam_info, items_boxes)

def process_results_from_yolo(frames: dict, batch_results: dict, cameras: dict, cam_names: list, classes: list, executor: ThreadPoolExecutor) -> dict:
    yolo_results = {}

    for idx, cam_id in enumerate(cam_names):
        res = batch_results[idx]
        full_box_boxes, slots_boxes, items_boxes = [], [], []

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
                    if len(slots_boxes) < 5:
                        slots_boxes.append((score, pts))
                    else:
                        min_idx = min(range(5), key=lambda i: slots_boxes[i][0])
                        if score > slots_boxes[min_idx][0]:
                            slots_boxes[min_idx] = (score, pts)
                            
                elif cls_id == 9:
                    if len(full_box_boxes) == 0:
                        full_box_boxes.append((score, pts))
                    else:
                        if score > full_box_boxes[0][0]:
                            full_box_boxes[0] = (score, pts)
                else:
                    item_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                    items_boxes.append((item_name, pts))

        yolo_results[cam_id] = {
            "full_box_boxes": [pts for _, pts in full_box_boxes], # tối đa 1 
            "slots_boxes": [pts for _, pts in slots_boxes], # tối đa 5
            "items_boxes": items_boxes
        }

    futures = []
    for cam_name, detect_result in yolo_results.items():
        frame = frames.get(cam_name)
        cam_info = cameras.get(cam_name)
        if frame is None or cam_info is None:
            continue
        futures.append(
            executor.submit(process_single_camera, detect_result, frame, cam_info)
        )

    for future in futures:
        future.result()

    return frames
