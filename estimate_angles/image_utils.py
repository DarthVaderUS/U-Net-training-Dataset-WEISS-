import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from collections import deque

def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    return (img > 127).astype(np.uint8)

def extract_skeleton(mask):
    sk = skeletonize(mask > 0)
    return sk.astype(np.uint8)

def find_endpoints(skeleton):
    endpoints = []
    h, w = skeleton.shape
    for r in range(1, h-1):
        for c in range(1, w-1):
            if skeleton[r, c]:
                neigh = skeleton[r-1:r+2, c-1:c+2]
                if np.sum(neigh) == 2:
                    endpoints.append((r, c))
    return endpoints

def bfs_path(skeleton, start, goal):
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, bool)
    parent = {}
    q = deque([start])
    visited[start] = True
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            return path[::-1]
        for dr,dc in dirs:
            nr,nc = cur[0]+dr,cur[1]+dc
            if 0<=nr<h and 0<=nc<w and skeleton[nr,nc] and not visited[nr,nc]:
                visited[nr,nc] = True
                parent[(nr,nc)] = cur
                q.append((nr,nc))
    return []

def get_ordered_skeleton_path(skeleton):
    eps = find_endpoints(skeleton)
    if len(eps)<2:
        return np.empty((0,2))
    if len(eps)>2:
        distmap = distance_transform_edt(skeleton)
        p1 = max(eps, key=lambda p: distmap[p])
        p2 = max(eps, key=lambda p: np.hypot(p[0]-p1[0], p[1]-p1[1]))
    else:
        p1,p2 = eps
    path = bfs_path(skeleton, p1, p2)
    return np.array(path, int)

def smooth_path(path, window=21, polyorder=3):
    """对(r,c)坐标做 Savitzky–Golay 滤波，获得子像素平滑曲线。"""
    if len(path) < window:
        return path.astype(float)
    r = savgol_filter(path[:,0].astype(float), window_length=window, polyorder=polyorder)
    c = savgol_filter(path[:,1].astype(float), window_length=window, polyorder=polyorder)
    return np.vstack((r,c)).T

def compute_total_turn_angle(path):
    total = 0.0
    for i in range(1, len(path)-1):
        p0,p1,p2 = path[i-1],path[i],path[i+1]
        v1 = p1 - p0
        v2 = p2 - p1
        ang1 = np.arctan2(v1[0], v1[1])
        ang2 = np.arctan2(v2[0], v2[1])
        delta = ang2 - ang1
        delta = (delta + np.pi) % (2*np.pi) - np.pi
        total += delta
    return np.degrees(total)

def estimate_turn_angle(mask_path):
    mask = load_mask(mask_path)
    sk = extract_skeleton(mask)
    raw_path = get_ordered_skeleton_path(sk)
    if raw_path.shape[0]<3:
        raise ValueError("Skeleton path too short.")
    path = smooth_path(raw_path)
    return compute_total_turn_angle(path)
