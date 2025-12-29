import numpy as np
import cv2
import os
from pathlib import Path
import random
from predict_slot import predict_slot

def parse_yolo_obb_label(label_path):
    """
    Đọc file txt nhãn YOLO OBB
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    """
    full_box = None
    slots = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            
            class_id = int(parts[0])
            points = np.array([float(x) for x in parts[1:]]).reshape(4, 2)
            
            if class_id == 9:  # full_box
                full_box = points
            elif class_id == 0:  # slot
                slots.append(points)
    
    return full_box, slots

def rotate_box(box, angle, center=(0.5, 0.5)):
    """
    Xoay box quanh center với góc angle (radian)
    box: (4, 2) normalized coordinates
    """
    center = np.array(center)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Dịch về gốc, xoay, rồi dịch lại
    rotated = []
    for point in box:
        shifted = point - center
        rotated_point = rot_matrix @ shifted
        rotated.append(rotated_point + center)
    
    return np.array(rotated)

def denormalize_box(box, img_width=640, img_height=640):
    """Chuyển từ normalized (0-1) sang pixel coordinates"""
    denorm = box.copy()
    denorm[:, 0] *= img_width
    denorm[:, 1] *= img_height
    return denorm.astype(np.int32)

def draw_box(img, box, color, thickness=2, label=None):
    """Vẽ box lên ảnh"""
    pts = box.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    
    if label:
        center = box.mean(axis=0).astype(int)
        cv2.putText(img, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)

def test_predict_slot(label_path, layout_type=12, output_dir='check', num_rotations=12, max_hidden_slots=3):
    """
    Test hàm predict_slot với nhiều trường hợp
    
    Args:
        label_path: Đường dẫn đến file label
        layout_type: 12 hoặc 34
        output_dir: Thư mục output
        num_rotations: Số góc xoay muốn test (mặc định 12)
        max_hidden_slots: Số slot tối đa muốn ẩn (mặc định 3)
    """
    # Tạo thư mục output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Đọc nhãn gốc
    full_box_orig, slots_orig = parse_yolo_obb_label(label_path)
    
    if full_box_orig is None or len(slots_orig) != 5:
        print(f"[ERROR] File {label_path} không hợp lệ (cần 1 full_box và 5 slots)")
        return
    
    file_name = Path(label_path).stem
    print(f"[INFO] Đọc được {len(slots_orig)} slots từ {file_name}")
    
    # Danh sách các góc xoay (độ)
    angle_step = 360 // num_rotations
    rotation_angles = [i * angle_step for i in range(num_rotations)]
    
    # Danh sách số slot bị ẩn
    num_hidden_slots_list = list(range(1, max_hidden_slots + 1))
    
    test_case = 0
    
    for angle_deg in rotation_angles:
        angle_rad = np.radians(angle_deg)
        
        # Xoay full_box và slots
        full_box_rotated = rotate_box(full_box_orig, angle_rad)
        slots_rotated = [rotate_box(slot, angle_rad) for slot in slots_orig]
        
        for num_hidden in num_hidden_slots_list:
            test_case += 1
            
            # Chọn ngẫu nhiên các slot để ẩn
            hidden_indices = random.sample(range(5), num_hidden)
            visible_indices = [i for i in range(5) if i not in hidden_indices]
            
            # Tạo slots_boxes (chỉ chứa slots không bị ẩn)
            slots_boxes = [slots_rotated[i] for i in visible_indices]
            
            print(f"  [TEST] {file_name} - Góc {angle_deg}°, Ẩn {num_hidden} slots")
            
            # Gọi hàm predict_slot
            try:
                slot_mapping, best_match = predict_slot(
                    layout_type=layout_type,
                    full_box_boxes=[full_box_rotated],
                    slots_boxes=slots_boxes
                )
                
                # Vẽ kết quả
                img = np.ones((640, 640, 3), dtype=np.uint8) * 255
                
                # Vẽ layout_map đã transform (lần khớp nhất) - màu xám nhạt (không label)
                slot_keys = ['slot_1', 'slot_2', 'slot_3', 'slot_4', 'slot_5'] if layout_type == 12 else ['slot_6', 'slot_7', 'slot_8', 'slot_9', 'slot_10']
                if best_match is not None:
                    for key in slot_keys:
                        if key in best_match:
                            layout_slot_transformed = best_match[key]
                            layout_slot_px = denormalize_box(layout_slot_transformed)
                            draw_box(img, layout_slot_px, (200, 200, 200), thickness=1)
                
                # Vẽ full_box rotated (màu xám đậm)
                full_box_px = denormalize_box(full_box_rotated)
                draw_box(img, full_box_px, (128, 128, 128), thickness=3)
                
                # Vẽ slots đầu vào (màu xanh dương) - không label
                for i, slot in enumerate(slots_boxes):
                    slot_px = denormalize_box(slot)
                    draw_box(img, slot_px, (255, 0, 0), thickness=2)
                
                # Vẽ ground truth của slots bị ẩn (màu xanh lá) - có label số thứ tự
                for idx in hidden_indices:
                    slot_px = denormalize_box(slots_rotated[idx])
                    # Tính slot_id từ index (0-4 -> 1-5 hoặc 6-10)
                    slot_id_start = 1 if layout_type == 12 else 6
                    slot_id = slot_id_start + idx
                    draw_box(img, slot_px, (0, 255, 0), thickness=2, label=str(slot_id))
                
                # Vẽ kết quả dự đoán (màu đỏ/cam) - có label slot_id
                for slot_id, (det_idx, points) in slot_mapping.items():
                    points_np = np.array(points)
                    points_px = denormalize_box(points_np)
                    
                    if det_idx == -1:  # Slot được dự đoán (thiếu)
                        draw_box(img, points_px, (0, 0, 255), thickness=3, label=str(slot_id))
                    else:  # Slot từ đầu vào (đã có)
                        draw_box(img, points_px, (255, 165, 0), thickness=2, label=str(slot_id))
                
                # Thêm text thông tin
                info_text = [
                    f"{file_name} - Layout {layout_type} - Rot {angle_deg}deg - Hide {num_hidden}",
                    f"Layout Map Transformed (Light Gray): Best match template",
                    f"Input (Blue): {len(slots_boxes)} | GT Hidden (Green): {len(hidden_indices)} | Predicted (Red): {len([v for v in slot_mapping.values() if v[0] == -1])}"
                ]
                
                y_pos = 30
                for text in info_text:
                    cv2.putText(img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                               0.45, (0, 0, 0), 1)
                    y_pos += 20
                
                # Lưu ảnh trực tiếp vào output_dir (không subfolder)
                output_path = os.path.join(output_dir, f"{file_name}_layout{layout_type}_rot{angle_deg}_hide{num_hidden}.jpg")
                cv2.imwrite(output_path, img)
                
            except Exception as e:
                print(f"    [ERROR] Failed: {str(e)}")
                import traceback
                traceback.print_exc()

def test_multiple_labels(label_paths, layout_type=12, output_base_dir='check', num_rotations=12, max_hidden_slots=3):
    """
    Test nhiều file label cùng lúc
    
    Args:
        label_paths: List đường dẫn các file label
        layout_type: 12 hoặc 34
        output_base_dir: Thư mục output gốc (tất cả ảnh lưu trực tiếp ở đây)
        num_rotations: Số góc xoay (mặc định 12)
        max_hidden_slots: Số slot tối đa muốn ẩn (mặc định 3)
    """
    # Tạo thư mục output
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    total_files = len(label_paths)
    print(f"[INFO] Bắt đầu test {total_files} files")
    print(f"[INFO] Mỗi file sẽ tạo {num_rotations * max_hidden_slots} test cases")
    print(f"[INFO] Tổng cộng: {total_files * num_rotations * max_hidden_slots} test cases")
    print("="*80)
    
    for idx, label_path in enumerate(label_paths, 1):
        print(f"\n[{idx}/{total_files}] Testing: {Path(label_path).name}")
        
        # Tất cả ảnh lưu trực tiếp vào output_base_dir (không tạo subfolder)
        test_predict_slot(
            label_path=label_path,
            layout_type=layout_type,
            output_dir=output_base_dir,
            num_rotations=num_rotations,
            max_hidden_slots=max_hidden_slots
        )
    
    print(f"\n{'='*80}")
    print(f"[COMPLETED] Đã test xong {total_files} files!")
    print(f"[LOCATION] Tất cả {total_files * num_rotations * max_hidden_slots} ảnh lưu tại: {output_base_dir}/")
    print('='*80)

if __name__ == "__main__":
    # ========== CẤU HÌNH - COPY LINK VÀO ĐÂY ==========
    data_for_layout_type_12 = [
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_1969.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2148.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2144.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2162.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2182.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2159.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2156.txt",
    ]

    data_for_layout_type_34 = [
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2167.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2127.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2111.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_2072.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_1995.txt",
        "/Users/thongbui.nd/Documents/Thong Bui/cuoi_ky_ai/create_dataset/labels/IMG_1989.txt",
    ]
    
    NUM_ROTATIONS = 5          # Số góc xoay (5 = mỗi 72°)
    MAX_HIDDEN_SLOTS = 1       # Ẩn tối đa bao nhiêu slot (1)
    OUTPUT_DIR = 'check'       # Thư mục lưu kết quả
    # ==================================================
    
    # Test layout type 12
    print("="*80)
    print("TEST LAYOUT TYPE 12")
    print("="*80)
    print("CẤU HÌNH TEST:")
    print(f"  - Số files: {len(data_for_layout_type_12)}")
    print(f"  - Layout type: 12")
    print(f"  - Số góc xoay: {NUM_ROTATIONS}")
    print(f"  - Ẩn tối đa: {MAX_HIDDEN_SLOTS} slots")
    print(f"  - Thư mục output: {OUTPUT_DIR}/")
    print(f"  - Tổng test cases: {len(data_for_layout_type_12) * NUM_ROTATIONS * MAX_HIDDEN_SLOTS}")
    print("="*80)
    
    test_multiple_labels(
        label_paths=data_for_layout_type_12,
        layout_type=12,
        output_base_dir=OUTPUT_DIR,
        num_rotations=NUM_ROTATIONS,
        max_hidden_slots=MAX_HIDDEN_SLOTS
    )
    
    # Test layout type 34
    print("\n" + "="*80)
    print("TEST LAYOUT TYPE 34")
    print("="*80)
    print("CẤU HÌNH TEST:")
    print(f"  - Số files: {len(data_for_layout_type_34)}")
    print(f"  - Layout type: 34")
    print(f"  - Số góc xoay: {NUM_ROTATIONS}")
    print(f"  - Ẩn tối đa: {MAX_HIDDEN_SLOTS} slots")
    print(f"  - Thư mục output: {OUTPUT_DIR}/")
    print(f"  - Tổng test cases: {len(data_for_layout_type_34) * NUM_ROTATIONS * MAX_HIDDEN_SLOTS}")
    print("="*80)
    
    test_multiple_labels(
        label_paths=data_for_layout_type_34,
        layout_type=34,
        output_base_dir=OUTPUT_DIR,
        num_rotations=NUM_ROTATIONS,
        max_hidden_slots=MAX_HIDDEN_SLOTS
    )
    

