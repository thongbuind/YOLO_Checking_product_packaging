import numpy as np
from dataclasses import dataclass, field

@dataclass
class SlotInfo:
    """
    Data model cho một slot
    - expected: item mong đợi
    - points: tọa độ 4 điểm của slot
    - state: trạng thái (empty/oke/wrong)
    - placed_item: item hiện tại trong slot
    """
    expected: str
    points: np.ndarray
    state: str = "empty"
    placed_item: str = None

    def update_state(self, new_state: str, placed_item: str = None):
        """Cập nhật state của slot"""
        self.state = new_state
        if placed_item:
            self.placed_item = placed_item

    def update_points(self, new_points: np.ndarray):
        """Cập nhật tọa độ của slot"""
        self.points = np.array(new_points, dtype=np.float32)

    def reset(self):
        """Reset slot về trạng thái ban đầu"""
        self.state = "empty"
        self.placed_item = None


@dataclass
class CameraInfo:
    """
    Data model cho một camera
    - cam_id: ID của camera (1-4)
    - expected_slots: config các slot mong đợi
    - state: trạng thái camera (waiting/checking/done/false)
    - slot_mapping: dict mapping slot_id -> SlotInfo
    - pending_result: kết quả đang chờ gửi
    - pending_timestamp: thời điểm bắt đầu pending
    """
    cam_id: int
    expected_slots: dict
    state: str = "waiting"
    slot_mapping: dict = None
    pending_result: dict = None
    pending_timestamp: float = 0
    
    def change_state(self, new_state: str):
        """Thay đổi state của camera"""
        self.state = new_state

    def initialize_slots(self, mapping: dict, current_boxes: list):
        """
        Khởi tạo slot_mapping lần đầu
        mapping: dict {slot_id: original_index}
        current_boxes: list of numpy arrays (4, 2)
        """
        self.slot_mapping = {}
        start = 1 if self.cam_id <= 2 else 6
        
        for slot_id in range(start, start + 5):
            orig_idx = mapping[slot_id]
            expected_item = self.expected_slots.get(f"slot_{slot_id}", "unknown")
            points = current_boxes[orig_idx]
            
            self.slot_mapping[slot_id] = SlotInfo(
                expected=expected_item,
                points=np.array(points, dtype=np.float32)
            )

    def update_slot(self, slot_id: int, state: str = None, placed_item: str = None, points: np.ndarray = None):
        """Cập nhật thông tin của một slot"""
        if not self.slot_mapping or slot_id not in self.slot_mapping:
            return
        
        slot = self.slot_mapping[slot_id]
        if state:
            slot.update_state(state, placed_item)
        if points is not None:
            slot.update_points(points)

    def get_slot(self, slot_id: int) -> SlotInfo:
        """Lấy thông tin của một slot"""
        if not self.slot_mapping:
            return None
        return self.slot_mapping.get(slot_id)

    def set_pending_result(self, success: bool, wrong_slots: list = None, snapshot: dict = None, timestamp: float = 0):
        """Lưu kết quả pending"""
        self.pending_result = {
            "success": success,
            "wrong_slots": wrong_slots or [],
            "snapshot": snapshot or {}
        }
        self.pending_timestamp = timestamp

    def clear_pending_result(self):
        """Xóa pending result"""
        self.pending_result = None
        self.pending_timestamp = 0

    def reset(self):
        """Reset camera về trạng thái ban đầu"""
        self.slot_mapping = None
        self.state = "waiting"
        self.pending_result = None
        self.pending_timestamp = 0

    def get_snapshot(self) -> dict:
        """Lấy snapshot của state hiện tại"""
        if not self.slot_mapping:
            return {}
        return {sid: info.state for sid, info in self.slot_mapping.items()}

    def get_required_slot_ids(self) -> list:
        """Trả về danh sách slot ID bắt buộc"""
        if self.cam_id == 1:
            return [1, 2, 3]
        elif self.cam_id == 2:
            return [4, 5]
        elif self.cam_id == 3:
            return [6, 7, 8]
        elif self.cam_id == 4:
            return [9, 10]
        return []
    