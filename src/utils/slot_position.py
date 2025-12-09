import numpy as np
from itertools import combinations

def parse_obb_label(line):
    """Đọc label OBB và tính center"""
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4
    
    return [class_id, center_x, center_y]

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

# Xác định thứ tự slot
def slot_position(cam, boxes):
    """
    Nếu là cam 1,2 thì số thứ tự slot sẽ từ 1 -> 5. Nếu là cam 3,4 thì số thứ tự slot sẽ từ 6 -> 10.

    Vẽ line3 đi qua 3 điểm thẳng hàng, line2 đi qua 2 điểm còn lại.
    Kẻ vector connect(x,y) từ điểm giữa line3 sang line 2, vuông góc với line3.

    Sẽ có 4 trường hợp với line3
    Trường hợp 1: x dương y dương - slot 1(hoặc 6) gần góc trên bên phải hơn slot 2(hoặc 7) và 3(hoặc 8).
    Trường hợp 2: x dương y âm - slot 1(hoặc 6) gần gốc toạ độ nhất.
    Trường hợp 3: x âm y âm - slot 1(hoặc 6) gần góc dưới bên trái nhất.
    Trường hợp 4: x âm y dương - slot 1(hoặc 6) gần góc dưới bên phải nhất.

    Với line2 thì điểm nào gần slot 1(hoặc 6) hơn thì sẽ là slot 4(hoặc 9), điểm còn lại là slot 5(hoặc 10).
    """
    if cam in [1, 2]:
        start_slot = 1
    elif cam in [3, 4]:
        start_slot = 6
    
    centers = np.array([[b[1], b[2]] for b in boxes])

    three_idx = find_collinear_three(centers)
    two_idx = [i for i in range(5) if i not in three_idx]

    line3_pts = centers[three_idx]
    line2_pts = centers[two_idx]

    v_line3 = line3_pts[2] - line3_pts[0]
    perp = np.array([-v_line3[1], v_line3[0]])
    perp = perp / np.linalg.norm(perp)

    p_mid = line3_pts[1]
    vec = line2_pts.mean(axis=0) - p_mid
    vec_connect = perp * np.dot(vec, perp)

    dx, dy = vec_connect

    if dx > 0 and dy > 0:
        corner = np.array([1,0])
    elif dx > 0 and dy < 0:
        corner = np.array([0,0])   
    elif dx < 0 and dy < 0:
        corner = np.array([0,1])
    else:
        corner = np.array([1,1])

    dist3 = np.linalg.norm(line3_pts - corner, axis=1)
    line3_order = [three_idx[i] for i in np.argsort(dist3)]

    slot1_pos = centers[line3_order[0]]
    dist2 = np.linalg.norm(line2_pts - slot1_pos, axis=1)
    line2_order = [two_idx[i] for i in np.argsort(dist2)]

    result = {}
    slot_num = start_slot
    for i in line3_order:
        result[slot_num] = boxes[i][0]
        slot_num += 1
    for i in line2_order:
        result[slot_num] = boxes[i][0]
        slot_num += 1

    return result, vec_connect, p_mid
