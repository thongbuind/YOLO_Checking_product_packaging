import cv2
import os
import numpy as np
from utils.slot_position import slot_position

# -------------------------------
# Xoay ảnh + boxes
# -------------------------------
def rotate_image_and_boxes(img, boxes, angle_deg):
    H, W = img.shape[:2]
    center = (W//2, H//2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, M, (W, H))

    rotated_boxes = []
    for b in boxes:
        cx_pixel = b[1]*W
        cy_pixel = b[2]*H
        pt = np.array([cx_pixel, cy_pixel, 1])
        cx_rot, cy_rot = M @ pt
        rotated_boxes.append((b[0], cx_rot/W, cy_rot/H, b[3], b[4]))
    return rotated_img, rotated_boxes

# -------------------------------
# Vẽ slot lên ảnh
# -------------------------------
def draw_slots(img, boxes, slot_res_tuple):
    """
    Vẽ:
    - Các slot với số
    - 2 line (line3 3 điểm, line2 2 điểm)
    - Vector connect vuông góc từ line3 -> line2
    - In tọa độ vector connect
    - TH1-TH4
    """
    H, W = img.shape[:2]
    img_draw = img.copy()

    # unpack kết quả từ slot_locate
    slot_res, vec_connect, p_mid = slot_res_tuple

    # 1. Vẽ slot
    for slot_num, box_id in slot_res.items():
        for b in boxes:
            if b[0] == box_id:
                cx, cy = int(b[1]*W), int(b[2]*H)
                break
        cv2.circle(img_draw, (cx, cy), 10, (0,0,255), -1)
        cv2.putText(img_draw, str(slot_num), (cx+7, cy-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4, cv2.LINE_AA)

    # 2. Lấy index line3 và line2 dựa trên slot_res
    slot_to_idx = {b[0]: i for i, b in enumerate(boxes)}  # ✅ fix mapping
    line3_idx = [slot_to_idx[slot_res[i]] for i in [1,2,3]]
    line2_idx = [slot_to_idx[slot_res[i]] for i in [4,5]]

    # 3. Lấy tọa độ centers
    centers = np.array([[b[1]*W, b[2]*H] for b in boxes])
    line3_px = [tuple(centers[i].astype(int)) for i in line3_idx]
    line2_px = [tuple(centers[i].astype(int)) for i in line2_idx]

    # 4. Vẽ line3
    for i in range(2):
        cv2.line(img_draw, line3_px[i], line3_px[i+1], (255,0,0), 3)

    # 5. Vẽ line2
    cv2.line(img_draw, line2_px[0], line2_px[1], (0,255,0), 3)

    # 6. Vẽ vector connect
    pt1 = (int(p_mid[0]*W), int(p_mid[1]*H))
    pt2 = (int((p_mid + vec_connect)[0]*W), int((p_mid + vec_connect)[1]*H))
    cv2.arrowedLine(img_draw, pt1, pt2, (0,0,255), 3, tipLength=0.1)

    # 7. In tọa độ vector connect
    coord_text = f"dx={vec_connect[0]:.3f}, dy={vec_connect[1]:.3f}"
    cv2.putText(img_draw, coord_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    # 8. Xác định TH1-TH4 dựa trên vec_connect
    dx, dy = vec_connect
    if dx > 0 and dy > 0:
        case_text = "TH1"
    elif dx > 0 and dy < 0:
        case_text = "TH2"
    elif dx < 0 and dy < 0:
        case_text = "TH3"
    else:
        case_text = "TH4"

    cv2.putText(img_draw, case_text, (10, H-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4, cv2.LINE_AA)

    return img_draw

# -------------------------------
# Test nhiều ảnh xoay
# -------------------------------
def test_slots(img_path, data_test, angles_deg, save_dir="/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/slot_position/result_layer_2"):
    os.makedirs(save_dir, exist_ok=True)
    img_orig = cv2.imread(img_path)

    for angle in angles_deg:
        rot_img, rot_data = rotate_image_and_boxes(img_orig, data_test, angle)
        res = slot_position(rot_data)
        img = draw_slots(rot_img, rot_data, res)
        save_path = os.path.join(save_dir, f"rot_{int(angle)}.png")
        cv2.imwrite(save_path, img)
        print(f"Lưu ảnh: {save_path}")

# -------------------------------
# Data test
# -------------------------------
data_test = [
    (1, 0.386885, 0.591192, 0.096856, 0.164743),
    (2, 0.513577, 0.591192, 0.591192, 0.164743),
    (3, 0.634648, 0.587301, 0.076966, 0.166040),
    (4, 0.414991, 0.457906, 0.142690, 0.070697),
    (5, 0.590111, 0.455312, 0.136637, 0.049942),
]

img_path = "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/test_func/slot_position/7D5B98D3-3C64-46EC-80F1-C79C3E9C5DBB_1_102_o.jpeg"
angles = np.linspace(0, 360, 20, endpoint=False)
test_slots(img_path, data_test, angles)