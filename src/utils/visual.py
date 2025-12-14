import cv2
import numpy as np
from datetime import datetime
import time
from collections import deque

class FPSCalculator:
    def __init__(self, buffer_size=30):
        self.timestamps = deque(maxlen=buffer_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0] + 1e-6)

def create_debug_table(fps_values, sys_fps, table_width=480, table_height=960):
    table = np.zeros((table_height, table_width, 3), dtype=np.uint8)
    table[:] = (30, 30, 30)
    
    y_offset = 15
    line_height = 24
    padding_left = 15
    
    COLOR_TITLE   = (0, 255, 255)
    COLOR_HEADER  = (100, 200, 255)
    COLOR_TEXT    = (255, 255, 255)
    COLOR_VALUE   = (0, 255, 0)
    COLOR_WARNING = (0, 165, 255)
    COLOR_ERROR   = (0, 0, 255)
    COLOR_DIVIDER = (80, 80, 80)

    def draw_text(text, y, color=COLOR_TEXT, font_scale=0.5, thickness=1, bold=False):
        if bold: thickness = 2
        cv2.putText(table, text, (padding_left, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return y + line_height
    
    def draw_line(y, color=COLOR_DIVIDER):
        cv2.line(table, (padding_left, y), (table_width - padding_left, y), color, 1)
        return y + line_height

    def draw_section(title, y):
        y = draw_text(title, y, COLOR_TITLE, 0.65, bold=True)
        return draw_line(y - 8)

    # Header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y_offset = draw_text("ASSEMBLY CHECK SYSTEM", y_offset, COLOR_TITLE, 0.8, bold=True)
    y_offset = draw_text(current_time, y_offset, COLOR_HEADER, 0.45)
    y_offset = draw_line(y_offset)

    # System FPS
    y_offset = draw_section("SYSTEM STATUS", y_offset)
    fps_color = COLOR_VALUE if sys_fps > 18 else COLOR_WARNING if sys_fps > 10 else COLOR_ERROR
    y_offset = draw_text(f"System FPS: {sys_fps:.1f}", y_offset, fps_color, 0.6, bold=True)
    y_offset += 10

    # Camera FPS
    y_offset = draw_section("CAMERA FPS", y_offset)
    for cam_name in sorted(fps_values.keys()):
        fps = fps_values.get(cam_name, 0)
        color = COLOR_VALUE if fps > 20 else COLOR_WARNING if fps > 10 else COLOR_ERROR
        status = "OK" if fps > 20 else "SLOW" if fps > 10 else "CRITICAL"
        y_offset = draw_text(f"{cam_name.upper()}: {fps:5.1f} FPS [{status}]", y_offset, color, 0.52)

    y_offset += 10

    # Footer
    footer_y = table_height - 40
    cv2.line(table, (0, footer_y), (table_width, footer_y), COLOR_DIVIDER, 1)
    cv2.putText(table, "Press 'q' to quit", (padding_left, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HEADER, 1)

    return table

def make_grid(frames, fps_values, sys_fps):
    # Grid 2x2
    top = np.hstack((frames["cam_1"], frames["cam_2"]))
    bottom = np.hstack((frames["cam_3"], frames["cam_4"]))
    grid = np.vstack((top, bottom))

    # Debug table
    debug_table = create_debug_table(
        fps_values=fps_values,
        sys_fps=sys_fps,
        table_width=500,
        table_height=grid.shape[0]
    )

    # Ghép ngang
    combined = np.hstack((grid, debug_table))
    return combined

SLOT_COLORS = {
    "empty": (0, 255, 255),      # Vàng - Slot trống
    "oke": (0, 255, 0),      # Xanh lá - Slot có item đúng
    "wrong": (0, 0, 255),    # Đỏ - Slot có item sai
    "default": (128, 128, 128) # Xám - State không xác định
}

# Màu item
ITEM_COLOR = (255, 255, 0)  # Vàng

# Màu camera state text
CAM_STATE_COLORS = {
    "done": (0, 255, 0),       # Xanh lá - Tất cả slots OK
    "false": (0, 0, 255),      # Đỏ - Có ít nhất 1 slot sai
    "checking": (0, 255, 255), # Vàng - Đang kiểm tra
    "waiting": (128, 128, 128) # Xám - Chờ slots xuất hiện
}

# Cấu hình text
CAM_LABEL_COLOR = (0, 255, 255)
SLOT_LINE_THICKNESS = 3
ITEM_LINE_THICKNESS = 2
TEXT_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_SCALE_SMALL = 0.5
TEXT_SCALE_LARGE = 0.8

# Cấu hình overlay (nền mờ)
OVERLAY_COLOR = (0, 0, 0)      # Đen
OVERLAY_ALPHA = 0.5            # Độ trong suốt (0-1)
OVERLAY_PADDING = 10           # Khoảng cách padding

def draw_visualization(frame, cam_info, items_boxes):
    _draw_all_slots(frame, cam_info)
    _draw_all_items(frame, items_boxes)
    _draw_camera_info(frame, cam_info)
    _draw_slot_status_table(frame, cam_info)

def _draw_all_slots(frame, cam_info):
    """Vẽ tất cả slots từ cam_info.slots_list"""
    for slot_id, slot in cam_info.slots_list.items():
        points = slot.get_points()
        if points is None:
            continue
        
        pts = np.int32(points)
        color = SLOT_COLORS.get(slot.state, SLOT_COLORS["default"])
        
        # Vẽ box
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=SLOT_LINE_THICKNESS)
        
        # Vẽ label
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        cv2.putText(
            frame,
            f"slot_{slot_id}",
            (center_x - 30, center_y),
            TEXT_FONT,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS
        )

def _draw_all_items(frame, items_boxes):
    """Vẽ tất cả items"""
    for item_name, points in items_boxes:
        pts = np.int32(points)
        
        # Vẽ box
        cv2.polylines(frame, [pts], isClosed=True, color=ITEM_COLOR, thickness=ITEM_LINE_THICKNESS)
        
        # Vẽ label
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        cv2.putText(
            frame,
            item_name,
            (center_x - 30, center_y - 10),
            TEXT_FONT,
            TEXT_SCALE,
            ITEM_COLOR,
            TEXT_THICKNESS
        )

def _draw_camera_info(frame, cam_info):
    """Vẽ thông tin camera (label + state) - Màu text theo state"""
    cam_label = _get_camera_label(cam_info.slot_will_be_checked)
    text_color = CAM_STATE_COLORS.get(cam_info.state, CAM_STATE_COLORS["waiting"])
    text = f"{cam_label}: {cam_info.state.upper()}"
    
    # Tính kích thước text
    (text_width, text_height), baseline = cv2.getTextSize(
        text, TEXT_FONT, TEXT_SCALE_LARGE, TEXT_THICKNESS
    )
    
    # Vẽ nền mờ
    x1, y1 = 10 - OVERLAY_PADDING, 30 - text_height - OVERLAY_PADDING
    x2, y2 = 10 + text_width + OVERLAY_PADDING, 30 + baseline + OVERLAY_PADDING
    _draw_overlay(frame, x1, y1, x2, y2)
    
    # Vẽ text
    cv2.putText(
        frame,
        text,
        (10, 30),
        TEXT_FONT,
        TEXT_SCALE_LARGE,
        text_color,
        TEXT_THICKNESS
    )

def _draw_slot_status_table(frame, cam_info):
    """Vẽ bảng trạng thái các slot cần check - Mỗi slot màu riêng theo state"""
    if cam_info.get_state() not in ["checking", "done", "false"]:
        return
    
    # Tính kích thước nền cho toàn bộ bảng
    slot_count = len(cam_info.slot_will_be_checked)
    if slot_count == 0:
        return
    
    # Tính kích thước text mẫu để ước lượng
    sample_text = "S10: wrong"
    (text_width, text_height), baseline = cv2.getTextSize(
        sample_text, TEXT_FONT, TEXT_SCALE_SMALL, TEXT_THICKNESS
    )
    
    # Vẽ nền mờ cho toàn bộ bảng
    x1 = 10 - OVERLAY_PADDING
    y1 = 60 - text_height - OVERLAY_PADDING
    x2 = 10 + text_width + OVERLAY_PADDING * 2
    y2 = 60 + (25 * slot_count) + OVERLAY_PADDING
    _draw_overlay(frame, x1, y1, x2, y2)
    
    # Vẽ từng slot
    y_offset = 60
    for slot_id in cam_info.slot_will_be_checked:
        slot = cam_info.get_slot(slot_id)
        if slot is None:
            continue
        
        # Mỗi slot có màu riêng dựa trên state của nó
        text_color = SLOT_COLORS.get(slot.state, SLOT_COLORS["default"])
        
        cv2.putText(
            frame,
            f"S{slot_id}: {slot.state}",
            (10, y_offset),
            TEXT_FONT,
            TEXT_SCALE_SMALL,
            text_color,
            TEXT_THICKNESS
        )
        y_offset += 25

def _get_camera_label(slot_will_be_checked):
    """Xác định camera label dựa trên slots cần check"""
    slots = set(slot_will_be_checked)
    
    if 1 in slots and 4 not in slots:
        return "CAM_1"
    elif 1 in slots and 4 in slots:
        return "CAM_2"
    elif 6 in slots and 9 not in slots:
        return "CAM_3"
    elif 6 in slots and 9 in slots:
        return "CAM_4"
    else:
        return "CAM_UNKNOWN"

def _draw_overlay(frame, x1, y1, x2, y2):
    """Vẽ nền mờ (overlay) tại vị trí chỉ định"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLOR, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
