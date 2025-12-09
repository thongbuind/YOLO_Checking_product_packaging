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

def create_debug_table(fps_values, detection_results, sys_fps,
                       table_width=480, table_height=960):
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

    # Detection Summary
    y_offset = draw_section("DETECTION SUMMARY", y_offset)
    total_slots = total_items = 0
    for cam_name in sorted(detection_results.keys()):
        det = detection_results[cam_name]
        n_slot = len(det.get('slot_boxes', []))
        n_item = len(det.get('item_boxes', []))
        total_slots += n_slot
        total_items += n_item
        y_offset = draw_text(f"{cam_name}:", y_offset, COLOR_HEADER, 0.52, bold=True)
        y_offset = draw_text(f"  Slots : {n_slot}/5", y_offset, COLOR_VALUE if n_slot==5 else COLOR_WARNING, 0.48)
        y_offset = draw_text(f"  Items : {n_item}", y_offset, COLOR_VALUE, 0.48)
    
    y_offset = draw_line(y_offset - 5)
    y_offset = draw_text(f"TOTAL → Slots: {total_slots} | Items: {total_items}", y_offset, COLOR_TITLE, 0.55, bold=True)
    y_offset += 10

    # Items Detected (hỗ trợ OBB)
    y_offset = draw_section("ITEMS DETECTED", y_offset)
    item_counts = {}
    has_item = False
    for cam_name in sorted(detection_results.keys()):
        items = detection_results[cam_name].get('item_boxes', [])
        if items:
            has_item = True
            y_offset = draw_text(f"{cam_name}:", y_offset, COLOR_HEADER, 0.52, bold=True)
            for item_name, box_data in items[:7]:
                try:
                    if isinstance(box_data, (list, tuple, np.ndarray)):
                        pts = np.array(box_data).reshape(-1, 2)
                        center = pts.mean(axis=0).astype(int)
                        pos = f"({center[0]},{center[1]})"
                    else:
                        pos = "(?,?)"
                except:
                    pos = "(error)"
                y_offset = draw_text(f"  {item_name} @ {pos}", y_offset, COLOR_VALUE, 0.45)
                item_counts[item_name] = item_counts.get(item_name, 0) + 1

    if not has_item:
        y_offset = draw_text("No items detected", y_offset, COLOR_WARNING, 0.52)

    # Item Statistics Bar
    if item_counts:
        y_offset += 10
        y_offset = draw_section("ITEM STATISTICS", y_offset)
        max_count = max(item_counts.values()) if item_counts else 1
        for item_name, count in sorted(item_counts.items(), key=lambda x: -x[1]):
            bar_len = int(180 * count / max_count)
            cv2.rectangle(table, (padding_left + 120, y_offset - 14),
                         (padding_left + 120 + bar_len, y_offset - 4), COLOR_VALUE, -1)
            y_offset = draw_text(f"{item_name}: {count}", y_offset, COLOR_TEXT, 0.48)

    # Alerts
    y_offset += 15
    y_offset = draw_section("ALERTS", y_offset)
    alerts = []
    for cam_name, fps in fps_values.items():
        if fps < 10:
            alerts.append(f"{cam_name} FPS quá thấp: {fps:.1f}")
    if total_slots == 0:
        alerts.append("Không phát hiện slot nào")
    for cam_name, det in detection_results.items():
        if len(det.get('slot_boxes', [])) not in (0, 5):
            alerts.append(f"{cam_name} chỉ thấy {len(det.get('slot_boxes', []))} slot")

    if alerts:
        for a in alerts[:7]:
            y_offset = draw_text(a, y_offset, COLOR_ERROR, 0.48, bold=True)
    else:
        y_offset = draw_text("Tất cả hệ thống ổn định", y_offset, COLOR_VALUE, 0.55)

    # Footer
    footer_y = table_height - 40
    cv2.line(table, (0, footer_y), (table_width, footer_y), COLOR_DIVIDER, 1)
    cv2.putText(table, "Press 'q' to quit", (padding_left, footer_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HEADER, 1)

    return table

def make_grid(frames, fps_values, detection_results, sys_fps):
    # Grid 2x2
    top = np.hstack((frames["cam_1"], frames["cam_2"]))
    bottom = np.hstack((frames["cam_3"], frames["cam_4"]))
    grid = np.vstack((top, bottom))

    # Debug table
    debug_table = create_debug_table(
        fps_values=fps_values,
        detection_results=detection_results or {},
        sys_fps=sys_fps,
        table_width=500,
        table_height=grid.shape[0]
    )

    # Ghép ngang
    combined = np.hstack((grid, debug_table))
    return combined

