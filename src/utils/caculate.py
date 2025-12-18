import numpy as np
from itertools import combinations
from shapely.geometry import Polygon

def fit_line_pca(points):
    pts = np.array(points)
    c = pts.mean(axis=0)
    shifted = pts - c
    cov = np.cov(shifted.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    d = eigvecs[:, np.argmax(eigvals)]
    return c, d / np.linalg.norm(d)

# Tìm 3 điểm thẳng hàng nhất
def find_collinear_three(points):
    pts = np.array(points)
    best_group = None
    best_error = float("inf")
    for comb in combinations(range(5), 3):
        subset = pts[list(comb)]
        c, d = fit_line_pca(subset)
        perp = np.array([-d[1], d[0]])
        shifted = subset - c
        dists = np.abs(np.dot(shifted, perp))
        total_error = dists.sum()
        if total_error < best_error:
            best_error = total_error
            best_group = list(comb)
    return best_group

# Tính diện tích bounding box
def bbox_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Tính phần diện tích chung nhau giữa 2 bounding box
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
