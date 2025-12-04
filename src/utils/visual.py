import cv2
import numpy as np
from datetime import datetime
from collections import deque
import time

class FPSCalculator:
    def __init__(self, buffer_size=30):
        self.timestamps = deque(maxlen=buffer_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0] + 1e-6)

# -----------------------------------------------------------------
# Debug Table - Hiển thị toàn bộ thông tin hệ thống
# -----------------------------------------------------------------

def create_debug_table(fps_values, detection_results, slots, sys_fps, 
                       table_width=450, table_height=960):
    """
    Tạo bảng debug với toàn bộ thông tin hệ thống
    
    Args:
        fps_values: dict {cam_name: fps_value}
        detection_results: dict {cam_name: {'slot_boxes': [], 'item_boxes': []}}
        slots: list [(slot_id, xyxy), ...]
        sys_fps: float - system FPS
        table_width: width của bảng
        table_height: height của bảng (bằng height của grid 2x2)
    """
    # Tạo canvas đen
    table = np.zeros((table_height, table_width, 3), dtype=np.uint8)
    table[:] = (30, 30, 30)  # Background màu xám đậm
    
    y_offset = 15
    line_height = 22
    padding_left = 15
    
    # Colors
    COLOR_TITLE = (0, 255, 255)      # Vàng
    COLOR_HEADER = (100, 200, 255)   # Xanh nhạt
    COLOR_TEXT = (255, 255, 255)     # Trắng
    COLOR_VALUE = (0, 255, 0)        # Xanh lá
    COLOR_WARNING = (0, 165, 255)    # Cam
    COLOR_ERROR = (0, 0, 255)        # Đỏ
    COLOR_DIVIDER = (80, 80, 80)     # Xám
    
    def draw_text(text, y, color=COLOR_TEXT, font_scale=0.5, thickness=1, bold=False):
        """Vẽ text lên bảng"""
        if bold:
            thickness = 2
        cv2.putText(table, text, (padding_left, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return y + line_height
    
    def draw_line(y, color=COLOR_DIVIDER):
        """Vẽ đường kẻ ngang"""
        cv2.line(table, (padding_left, y), (table_width - padding_left, y), color, 1)
        return y + line_height
    
    def draw_section_title(title, y):
        """Vẽ tiêu đề section"""
        y = draw_text(title, y, COLOR_TITLE, font_scale=0.6, bold=True)
        return draw_line(y - 10)
    
    # -----------------------------------------------------------------
    # HEADER - Thời gian và System Info
    # -----------------------------------------------------------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y_offset = draw_text("DEBUG MONITOR", y_offset, COLOR_TITLE, 0.7, bold=True)
    y_offset = draw_text(current_time, y_offset, COLOR_HEADER, 0.4)
    y_offset = draw_line(y_offset)
    
    # System FPS
    y_offset = draw_section_title("SYSTEM STATUS", y_offset)
    fps_color = COLOR_VALUE if sys_fps > 15 else COLOR_WARNING if sys_fps > 10 else COLOR_ERROR
    y_offset = draw_text(f"System FPS: {sys_fps:.1f}", y_offset, fps_color, 0.55, bold=True)
    y_offset += 10
    
    # -----------------------------------------------------------------
    # CAMERA FPS
    # -----------------------------------------------------------------
    y_offset = draw_section_title("CAMERA FPS", y_offset)
    
    for cam_name in sorted(fps_values.keys()):
        fps = fps_values.get(cam_name, 0)
        fps_color = COLOR_VALUE if fps > 20 else COLOR_WARNING if fps > 10 else COLOR_ERROR
        status = "OK" if fps > 20 else "SLOW" if fps > 10 else "CRITICAL"
        
        y_offset = draw_text(f"{cam_name}: {fps:5.1f} FPS [{status}]", 
                            y_offset, fps_color, 0.5)
    
    y_offset += 10
    
    # -----------------------------------------------------------------
    # DETECTION SUMMARY
    # -----------------------------------------------------------------
    y_offset = draw_section_title("DETECTION SUMMARY", y_offset)
    
    total_slots = 0
    total_items = 0
    
    for cam_name in sorted(detection_results.keys()):
        det = detection_results[cam_name]
        num_slots = len(det.get('slot_boxes', []))
        num_items = len(det.get('item_boxes', []))
        
        total_slots += num_slots
        total_items += num_items
        
        y_offset = draw_text(f"{cam_name}:", y_offset, COLOR_HEADER, 0.5, bold=True)
        y_offset = draw_text(f"  Slots: {num_slots}", y_offset, COLOR_VALUE, 0.45)
        y_offset = draw_text(f"  Items: {num_items}", y_offset, COLOR_VALUE, 0.45)
    
    y_offset = draw_line(y_offset - 5)
    y_offset = draw_text(f"TOTAL - Slots: {total_slots} | Items: {total_items}", 
                        y_offset, COLOR_TITLE, 0.5, bold=True)
    y_offset += 10
    
    # -----------------------------------------------------------------
    # SLOT POSITION STATUS
    # -----------------------------------------------------------------
    y_offset = draw_section_title("SLOT POSITIONS", y_offset)
    
    if len(slots) > 0:
        y_offset = draw_text(f"Active Slots: {len(slots)}", y_offset, COLOR_VALUE, 0.5)
        
        for i, (slot_id, slot_xyxy) in enumerate(slots[:5]):  # Hiển thị tối đa 5 slots
            x1, y1, x2, y2 = map(int, slot_xyxy)
            slot_info = f"  #{slot_id}: ({x1},{y1})-({x2},{y2})"
            y_offset = draw_text(slot_info, y_offset, COLOR_TEXT, 0.4)
    else:
        y_offset = draw_text("No slots detected", y_offset, COLOR_WARNING, 0.5)
    
    y_offset += 10
    
    # -----------------------------------------------------------------
    # ITEMS DETECTED PER CAMERA
    # -----------------------------------------------------------------
    y_offset = draw_section_title("ITEMS DETECTED", y_offset)
    
    item_counts = {}
    
    for cam_name in sorted(detection_results.keys()):
        det = detection_results[cam_name]
        item_boxes = det.get('item_boxes', [])
        
        if len(item_boxes) > 0:
            y_offset = draw_text(f"{cam_name}:", y_offset, COLOR_HEADER, 0.5, bold=True)
            
            for item_name, item_xyxy in item_boxes[:8]:  # Giới hạn hiển thị
                x1, y1, x2, y2 = map(int, item_xyxy)
                y_offset = draw_text(f"  {item_name} @ ({x1},{y1})", 
                                    y_offset, COLOR_VALUE, 0.4)
                
                # Count items
                item_counts[item_name] = item_counts.get(item_name, 0) + 1
    
    if not item_counts:
        y_offset = draw_text("No items detected", y_offset, COLOR_WARNING, 0.5)
    
    y_offset += 10
    
    # -----------------------------------------------------------------
    # ITEM STATISTICS
    # -----------------------------------------------------------------
    if item_counts:
        y_offset = draw_section_title("ITEM STATISTICS", y_offset)
        
        for item_name, count in sorted(item_counts.items(), key=lambda x: x[1], reverse=True):
            bar_length = min(int(count * 20), table_width - padding_left * 2 - 100)
            
            # Draw bar
            bar_y = y_offset - 8
            cv2.rectangle(table, 
                         (padding_left + 100, bar_y - 10), 
                         (padding_left + 100 + bar_length, bar_y + 2),
                         COLOR_VALUE, -1)
            
            y_offset = draw_text(f"{item_name}: {count}", y_offset, COLOR_TEXT, 0.45)
    
    y_offset += 10
    
    # -----------------------------------------------------------------
    # WARNINGS & ALERTS
    # -----------------------------------------------------------------
    y_offset = draw_section_title("ALERTS", y_offset)
    
    alerts = []
    
    # Check low FPS
    for cam_name, fps in fps_values.items():
        if fps < 10:
            alerts.append(f"! {cam_name} FPS too low: {fps:.1f}")
    
    # Check missing slots
    if total_slots == 0:
        alerts.append("! No slots detected")
    
    # Check slot count per camera
    for cam_name, det in detection_results.items():
        num_slots = len(det.get('slot_boxes', []))
        if num_slots > 0 and num_slots != 5:
            alerts.append(f"! {cam_name} slot count: {num_slots}/5")
    
    if alerts:
        for alert in alerts[:6]:  # Giới hạn 6 alerts
            y_offset = draw_text(alert, y_offset, COLOR_ERROR, 0.45, bold=True)
    else:
        y_offset = draw_text("All systems nominal", y_offset, COLOR_VALUE, 0.5)
    
    # -----------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------
    footer_y = table_height - 30
    cv2.line(table, (0, footer_y), (table_width, footer_y), COLOR_DIVIDER, 1)
    # cv2.putText(table, "Press 'q' to quit", 
    #            (padding_left, footer_y + 20),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HEADER, 1)
    
    return table


def make_grid(frames, fps_values, detection_results, slots, sys_fps):
    """
    Tạo grid 2x2 với debug table ở bên phải
    
    Args:
        frames: dict {cam_name: frame}
        fps_values: dict {cam_name: fps}
        detection_results: dict {cam_name: detection_data}
        slots: list of slots
        sys_fps: system fps
    
    Returns:
        Combined image (grid + debug table)
    """
    # Tạo grid 2x2 từ 4 cameras
    top = np.hstack((frames["cam1"], frames["cam2"]))
    bottom = np.hstack((frames["cam3"], frames["cam4"]))
    grid = np.vstack((top, bottom))
    
    # Tạo debug table với height bằng grid
    debug_table = create_debug_table(
        fps_values=fps_values,
        detection_results=detection_results,
        slots=slots,
        sys_fps=sys_fps,
        table_width=450,
        table_height=grid.shape[0]
    )
    
    # Ghép grid và debug table theo chiều ngang
    combined = np.hstack((grid, debug_table))
    
    return combined
