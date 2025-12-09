import numpy as np
import cv2

def calculate_iou_obb(box1_points, box2_points):
    """
    Tính IoU giữa 2 OBB (Oriented Bounding Box)
    box1_points, box2_points: numpy array shape (4, 2) - 4 điểm góc
    """
    try:
        box1 = np.float32(box1_points)
        box2 = np.float32(box2_points)
        
        # Tính diện tích từng box
        area1 = cv2.contourArea(box1)
        area2 = cv2.contourArea(box2)
        
        if area1 < 1 or area2 < 1:
            return 0.0
        
        # Tính diện tích giao
        retval, intersecting_region = cv2.intersectConvexConvex(box1, box2)
        
        if retval < 0 or intersecting_region is None:
            return 0.0
        
        intersection_area = cv2.contourArea(intersecting_region)
        
        if intersection_area <= 0:
            return 0.0
        
        # IoU = intersection / union
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    except Exception as e:
        print(f"[ERROR] calculate_iou_obb failed: {e}")
        return 0.0

def has_collision(item_points, slot_points):
    """
    Kiểm tra xem item có va chạm với slot không
    Chỉ cần có overlap bất kỳ (intersection > 0)
    """
    try:
        item_box = np.float32(item_points)
        slot_box = np.float32(slot_points)
        
        # cv2.intersectConvexConvex trả về (retval, intersectingRegion)
        retval, intersecting_region = cv2.intersectConvexConvex(item_box, slot_box)
        
        # retval < 0: không có giao
        # retval >= 0: có giao, intersecting_region là polygon
        if retval < 0 or intersecting_region is None:
            return False
        
        # Tính diện tích giao
        intersection_area = cv2.contourArea(intersecting_region)
        
        return intersection_area > 0
    
    except Exception as e:
        print(f"[ERROR] has_collision failed: {e}")
        return False

def is_item_in_slot(item_points, slot_points, threshold=0.8):
    """
    Kiểm tra xem item có nằm trong slot không
    Điều kiện: >= 80% diện tích item nằm trong slot
    
    Args:
        item_points: numpy array (4, 2) - 4 điểm góc của item
        slot_points: numpy array (4, 2) - 4 điểm góc của slot
        threshold: ngưỡng (mặc định 0.8 = 80%)
    
    Returns:
        float: tỷ lệ item nằm trong slot (0.0 -> 1.0)
    """
    try:
        item_box = np.float32(item_points)
        slot_box = np.float32(slot_points)
        
        # Tính diện tích item
        item_area = cv2.contourArea(item_box)
        
        if item_area < 100:  # Item quá nhỏ, bỏ qua
            return 0.0
        
        # Tính diện tích giao
        retval, intersecting_region = cv2.intersectConvexConvex(item_box, slot_box)
        
        if retval < 0 or intersecting_region is None:
            return 0.0
        
        intersection_area = cv2.contourArea(intersecting_region)
        
        if intersection_area <= 0:
            return 0.0
        
        # Tỷ lệ item nằm trong slot
        ratio = intersection_area / item_area
        
        return ratio
    
    except Exception as e:
        print(f"[ERROR] is_item_in_slot failed: {e}")
        return 0.0
    
def obb_to_xyxy(points):
    """
    Convert OBB (4 điểm) thành bounding box xyxy [x1, y1, x2, y2]
    Dùng cho visualization hoặc fallback
    """
    try:
        pts = np.array(points).reshape(-1, 2)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return [float(x1), float(y1), float(x2), float(y2)]
    except:
        return [0, 0, 0, 0]


def get_obb_center(points):
    """
    Lấy tâm của OBB
    """
    try:
        pts = np.array(points).reshape(-1, 2)
        center = pts.mean(axis=0)
        return center
    except:
        return np.array([0, 0])
    