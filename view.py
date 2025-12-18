# -*- coding: utf-8 -*-
"""
OBS 虚拟摄像头 + OpenCV + YOLOv8 (Det Track + Pose ROI)
退出：按 '-' 或 ESC
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
# 参数
# ---------------------------
CAMERA_INDEX_PREFER = 0

REQUEST_WIDTH = 2560
REQUEST_HEIGHT = 1440

MAX_WINDOW_LONG_SIDE = 1280

# ---------------------------
# 模型选择
# ---------------------------
DET_MODEL_WEIGHTS = "yolov8n.pt"
POSE_MODEL_WEIGHTS = "yolov8n-pose.pt"

PERSON_CLASS_ID = 0
TRACKER_CFG = "bytetrack.yaml"

# ---------------------------
# 推理与性能参数
# ---------------------------
DET_CONF = 0.33
DET_IOU = 0.55
MAX_DET = 30

DET_IMGSZ = 680
DET_INFER_MAX_LONG_SIDE = 1280  # 0 不缩放

POSE_CONF = 0.25
POSE_IMGSZ = 320
POSE_STRIDE = 1           # 每个 tid 至少隔 N 帧更新一次
POSE_MAX_PEOPLE = 20

# 关键点最低置信度（侧脸放宽一点）
KP_MIN_CONF = 0.10

# 头框 EMA 平滑系数
HEAD_EMA_ALPHA = 0.18

# 轨迹丢失后保留多少帧再清理
STALE_FRAMES = 60

# 新头框与旧头框差太大则重置平滑
IOU_RESET_THRESH = 0.12

CROP_PAD_RATIO = 0.08


# ---------------------------
# 摄像头相关
# ---------------------------
def open_camera(prefer_index: int = 0):
    """尝试打开摄像头（优先 prefer_index），失败则扫描 0~9；Windows 优先 DSHOW。"""
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(0)

    def try_open(idx: int):
        for be in backends:
            cap = cv2.VideoCapture(idx, be) if be != 0 else cv2.VideoCapture(idx)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, be
            cap.release()
        return None, None

    cap, be = try_open(prefer_index)
    if cap is not None:
        return cap, prefer_index, be

    for i in range(10):
        if i == prefer_index:
            continue
        cap, be = try_open(i)
        if cap is not None:
            return cap, i, be

    raise RuntimeError("无法打开 OBS 虚拟摄像头：请确认 OBS 已启动虚拟摄像头，或调整 camera index。")


def request_resolution(cap: cv2.VideoCapture, w: int, h: int):
    """向摄像头请求分辨率（可能被忽略），返回 CAP_PROP（不一定等于真实帧）。"""
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    try:
        cap.set(cv2.CAP_PROP_FPS, 60)
    except Exception:
        pass
    return cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


def warmup_read(cap: cv2.VideoCapture, n: int = 8):
    """读几帧让分辨率稳定。"""
    frame = None
    for _ in range(max(1, n)):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        raise RuntimeError("摄像头读帧失败。")
    return frame


# ---------------------------
# 工具函数
# ---------------------------
def get_device_and_half():
    """有 CUDA -> (0, True)，否则 ('cpu', False)。"""
    try:
        import torch
        if torch.cuda.is_available():
            return 0, True
    except Exception:
        pass
    return "cpu", False


def calc_display_size(w: int, h: int, max_long_side: int):
    """等比缩小窗口，不改变处理分辨率。"""
    long_side = max(w, h)
    if long_side <= max_long_side:
        return w, h
    scale = max_long_side / float(long_side)
    dw = max(320, int(w * scale))
    dh = max(240, int(h * scale))
    return dw, dh


def clamp_box_draw(x1, y1, x2, y2, w, h):
    """绘制用：限制到 [0, W-1]/[0, H-1]。"""
    x1 = float(max(0, min(w - 1, x1)))
    y1 = float(max(0, min(h - 1, y1)))
    x2 = float(max(0, min(w - 1, x2)))
    y2 = float(max(0, min(h - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_box_crop(x1, y1, x2, y2, w, h):
    """
    裁剪用：右下角切片是开区间，x2/y2 用 [0, W]/[0, H] 更稳，不容易切掉边缘。
    """
    x1 = float(max(0, min(w, x1)))
    y1 = float(max(0, min(h, y1)))
    x2 = float(max(0, min(w, x2)))
    y2 = float(max(0, min(h, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou(a, b):
    """a,b: [x1,y1,x2,y2]"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def ema_update(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * prev + alpha * cur


def maybe_resize_for_det(frame, max_long_side: int):
    """det/track 可选缩放推理。返回 (frame_det, scale) ，原图坐标 = det坐标 / scale"""
    if max_long_side is None or max_long_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    long_side = max(w, h)
    if long_side <= max_long_side:
        return frame, 1.0
    scale = max_long_side / float(long_side)
    dw = int(w * scale)
    dh = int(h * scale)
    frame_det = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
    return frame_det, scale


def crop_with_pad(frame, box_xyxy, pad_ratio: float):
    """按 box 裁剪并加 padding。返回 crop 和 (ox, oy) 偏移。"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    pad = pad_ratio * max(bw, bh)

    cx1 = x1 - pad
    cy1 = y1 - pad
    cx2 = x2 + pad
    cy2 = y2 + pad
    cx1, cy1, cx2, cy2 = clamp_box_crop(cx1, cy1, cx2, cy2, W, H)

    ix1, iy1, ix2, iy2 = map(int, [cx1, cy1, cx2, cy2])
    if ix2 <= ix1 + 1 or iy2 <= iy1 + 1:
        return None, 0, 0
    crop = frame[iy1:iy2, ix1:ix2].copy()
    return crop, ix1, iy1


def _kp_ok(kconf, idx, thr):
    if kconf is None:
        return True
    return float(kconf[idx]) >= thr


# ---------------------------
# ROI pose 里可能有多人，挑 ROI 主人
# ---------------------------
def pick_best_pose_index(pr, cw, ch):
    if getattr(pr, "boxes", None) is None or pr.boxes is None or len(pr.boxes) == 0:
        return 0
    try:
        b = pr.boxes.xyxy.cpu().numpy()
        c = pr.boxes.conf.cpu().numpy() if pr.boxes.conf is not None else None
    except Exception:
        return 0

    cx0, cy0 = cw * 0.5, ch * 0.5
    best_i, best_score = 0, -1e18
    for i, (x1, y1, x2, y2) in enumerate(b):
        bx = (x1 + x2) * 0.5
        by = (y1 + y2) * 0.5
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        dist2 = (bx - cx0) ** 2 + (by - cy0) ** 2
        conf = float(c[i]) if c is not None else 1.0
        score = -dist2 + 0.10 * area + 2000.0 * conf
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


# ---------------------------
# “更紧的整头框”（ROI 坐标系）
# - 核心改动：用 median 而不是 max，肩宽权重更小，且加上下限更紧/上限防爆
# ---------------------------
def head_box_full_from_pose(cw, ch, kxy, kconf=None):
    px1, py1, px2, py2 = 0.0, 0.0, float(cw), float(ch)
    pw, ph = px2 - px1, py2 - py1

    def get_pt(i):
        if not _kp_ok(kconf, i, KP_MIN_CONF):
            return None
        x, y = float(kxy[i][0]), float(kxy[i][1])
        margin = 12.0
        if x < px1 - margin or x > px2 + margin or y < py1 - margin or y > py2 + margin:
            return None
        return np.array([x, y], dtype=np.float32)

    # COCO keypoints:
    # 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder
    nose = get_pt(0)
    leye = get_pt(1)
    reye = get_pt(2)
    lear = get_pt(3)
    rear = get_pt(4)
    lsh = get_pt(5)
    rsh = get_pt(6)

    # --- 定位中心 & 顶部参考 y（先粗略）
    if lear is not None and rear is not None:
        center = (lear + rear) * 0.5
        top_ref_y = min(
            lear[1], rear[1],
            leye[1] if leye is not None else 1e9,
            reye[1] if reye is not None else 1e9,
            nose[1] if nose is not None else 1e9
        )
    elif leye is not None and reye is not None:
        center = (leye + reye) * 0.5
        top_ref_y = min(leye[1], reye[1], nose[1] if nose is not None else 1e9)
    elif (lear is not None) ^ (rear is not None):
        ear = lear if lear is not None else rear
        ref = nose if nose is not None else (leye if leye is not None else reye)
        if ref is not None:
            center = (ear + ref) * 0.5
            top_ref_y = min(ear[1], ref[1])
        else:
            center = ear
            top_ref_y = ear[1]
    elif nose is not None:
        center = nose
        top_ref_y = nose[1]
    else:
        center = np.array([pw * 0.5, ph * 0.18], dtype=np.float32)
        top_ref_y = float(ph * 0.10)

    cx, cy = float(center[0]), float(center[1])

    # --- 估计头宽：更紧、更稳（median + clip）
    cand = []
    if lear is not None and rear is not None:
        cand.append(float(np.linalg.norm(lear - rear)) * 1.55)   # 原 1.75
    if leye is not None and reye is not None:
        cand.append(float(np.linalg.norm(leye - reye)) * 2.80)   # 原 3.10
    if lsh is not None and rsh is not None:
        cand.append(float(np.linalg.norm(lsh - rsh)) * 0.38)     # 原 0.50（肩宽最容易拉爆）

    if cand:
        base_w = float(np.median(cand))  # 原 max(cand) 很容易变大
    else:
        base_w = 0.48 * pw               # 原 0.62 * pw

    # 高度更紧：避免吃到上半身
    base_h = 1.00 * base_w               # 原 1.12

    # 更紧下限 + 上限防爆（关键：原 min_h=0.42*ph 会直接把上半身吃进去）
    min_w = 0.32 * pw
    min_h = 0.22 * ph
    max_w = 0.68 * pw
    max_h = 0.52 * ph

    head_w = float(np.clip(base_w, min_w, max_w))
    head_h = float(np.clip(base_h, min_h, max_h))

    # --- 侧脸：单耳时轻推（小幅度，更精准）
    if (lear is not None) ^ (rear is not None):
        ear = lear if lear is not None else rear
        ref = nose if nose is not None else (leye if leye is not None else reye)
        if ref is not None:
            v = ear - ref
            n = float(np.linalg.norm(v)) + 1e-6
            v = v / n
            cx += float(v[0]) * 0.08 * head_w
            cy += float(v[1]) * 0.02 * head_h

    # 顶部参考更稳：鼻子存在时，允许略上探；单耳再略上提，避免侧脸“缩到眼部”
    if nose is not None:
        top_ref_y = min(float(top_ref_y), float(nose[1]) - 0.10 * head_h)
    if (lear is not None) ^ (rear is not None):
        top_ref_y = float(top_ref_y) - 0.03 * head_h

    # 头顶上抬：略加大（更贴头顶，同时底部不会太深）
    hy1 = float(top_ref_y) - 0.34 * head_h
    hy2 = hy1 + head_h

    hx1 = cx - 0.5 * head_w
    hx2 = cx + 0.5 * head_w

    return hx1, hy1, hx2, hy2


def head_box_fallback_full(px1, py1, px2, py2):
    """关键点不可用时的更紧 fallback（避免吃到上半身）。"""
    pw = px2 - px1
    ph = py2 - py1
    cx = (px1 + px2) / 2.0

    head_h = 0.30 * ph   # 原 0.40
    head_w = 0.52 * pw   # 原 0.62

    hx1 = cx - 0.5 * head_w
    hx2 = cx + 0.5 * head_w

    # 顶部稍低于人框顶，减少把上方空白/上半身吃进去的概率
    hy1 = py1 + 0.02 * ph
    hy2 = hy1 + head_h
    return hx1, hy1, hx2, hy2


# ---------------------------
# 主程序
# ---------------------------
def main():
    cap, used_index, used_backend = open_camera(CAMERA_INDEX_PREFER)
    print(f"[INFO] Camera opened. index={used_index}, backend={used_backend}")

    print(f"[INFO] Requesting resolution: {REQUEST_WIDTH}x{REQUEST_HEIGHT}")
    prop_w, prop_h = request_resolution(cap, REQUEST_WIDTH, REQUEST_HEIGHT)
    print(f"[INFO] CAP_PROP after set: {int(prop_w)}x{int(prop_h)} (may be inaccurate)")

    first_frame = warmup_read(cap, n=8)
    H0, W0 = first_frame.shape[:2]
    print(f"[INFO] REAL frame resolution: {W0}x{H0}")
    print("[INFO] Press '-' or ESC to quit.")

    device, half_ok = get_device_and_half()
    print(f"[INFO] Device: {device} | FP16: {half_ok}")

    det_model = YOLO(DET_MODEL_WEIGHTS)
    pose_model = YOLO(POSE_MODEL_WEIGHTS)

    print(f"[INFO] Det model:  {DET_MODEL_WEIGHTS} | imgsz={DET_IMGSZ} | tracker={TRACKER_CFG}")
    print(f"[INFO] Pose model: {POSE_MODEL_WEIGHTS} | pose_imgsz={POSE_IMGSZ} | stride={POSE_STRIDE}")

    win_name = "OBS + YOLOv8 (DetTrack + PoseROI) FULL-HEAD (TIGHT)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    disp_w, disp_h = calc_display_size(W0, H0, MAX_WINDOW_LONG_SIDE)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    print(f"[INFO] Window: {disp_w}x{disp_h} (frame stays {W0}x{H0})")

    # 只对“有稳定 tid”的人做 EMA 状态
    head_state = {}        # tid -> bbox(float32[4])
    last_seen = {}         # tid -> frame_idx
    last_pose_update = {}  # tid -> frame_idx

    frame_idx = 0
    fps = 0.0
    last_t = time.time()

    frame = first_frame

    while True:
        if frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read frame.")
                break

        H, W = frame.shape[:2]

        # det/track 可选缩放推理
        frame_det, det_scale = maybe_resize_for_det(frame, DET_INFER_MAX_LONG_SIDE)

        # 1) Det Track：人框 + tid
        results_det = det_model.track(
            frame_det,
            conf=DET_CONF,
            iou=DET_IOU,
            imgsz=DET_IMGSZ,
            device=device,
            half=half_ok,
            persist=True,
            verbose=False,
            classes=[PERSON_CLASS_ID],
            max_det=MAX_DET,
            tracker=TRACKER_CFG,
        )

        if not results_det or len(results_det) == 0:
            det = None
        else:
            det = results_det[0]

        tracks = {}        # tid(int) -> (person_box_xyxy, score)
        tracks_noid = []   # [(person_box_xyxy, score)] 没有稳定 tid 的人（不进 EMA）

        if det is not None and det.boxes is not None and len(det.boxes) > 0:
            boxes = det.boxes.xyxy
            confs = det.boxes.conf
            clss = det.boxes.cls

            ids = None
            if getattr(det.boxes, "id", None) is not None and det.boxes.id is not None:
                try:
                    ids = det.boxes.id
                except Exception:
                    ids = None

            boxes = boxes.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
            if ids is not None:
                try:
                    ids = ids.cpu().numpy().astype(int)
                except Exception:
                    ids = None

            for i, ((x1, y1, x2, y2), score, cls_id) in enumerate(zip(boxes, confs, clss)):
                if int(cls_id) != PERSON_CLASS_ID:
                    continue

                # 映射回原图坐标
                x1 /= det_scale
                y1 /= det_scale
                x2 /= det_scale
                y2 /= det_scale

                px1, py1, px2, py2 = clamp_box_draw(x1, y1, x2, y2, W, H)
                pbox = np.array([px1, py1, px2, py2], dtype=np.float32)

                if ids is None or i >= len(ids):
                    tracks_noid.append((pbox, float(score)))
                else:
                    tid = int(ids[i])
                    tracks[tid] = (pbox, float(score))
                    last_seen[tid] = frame_idx

        # 2) 决定本帧要对哪些（有 tid 的）人跑 pose（按 tid 隔帧 + 人多限制）
        need_pose_tids = []
        if tracks:
            def rank_key(item):
                tid, (b, s) = item
                area = float(max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])))
                return (area, s)

            sorted_tracks = sorted(tracks.items(), key=rank_key, reverse=True)

            for tid, (pbox, score) in sorted_tracks:
                last_upd = last_pose_update.get(tid, -10**9)
                if (tid not in head_state) or ((frame_idx - last_upd) >= max(1, POSE_STRIDE)):
                    need_pose_tids.append(tid)

            if len(need_pose_tids) > POSE_MAX_PEOPLE:
                keep = set([tid for tid, _ in sorted_tracks[:POSE_MAX_PEOPLE]])
                need_pose_tids = [tid for tid in need_pose_tids if tid in keep]

        # 2b) 没有 tid 的人：只在本帧做 pose（不做 EMA/不做 stride），数量也限制
        need_pose_noid = []
        if tracks_noid:
            tracks_noid_sorted = sorted(
                tracks_noid,
                key=lambda bs: float(max(0.0, (bs[0][2]-bs[0][0])) * max(0.0, (bs[0][3]-bs[0][1]))),
                reverse=True
            )
            need_pose_noid = tracks_noid_sorted[:min(POSE_MAX_PEOPLE, len(tracks_noid_sorted))]

        # 3) 对选中的人做 ROI pose（批量推理）
        # 3a) 有 tid
        if need_pose_tids:
            crops = []
            meta = []  # (tid, ox, oy, cw, ch)
            for tid in need_pose_tids:
                pbox, _ = tracks[tid]
                crop, ox, oy = crop_with_pad(frame, pbox, CROP_PAD_RATIO)
                if crop is None or crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                if cw < 10 or ch < 10:
                    continue
                crops.append(crop)
                meta.append((tid, ox, oy, cw, ch))

            if crops:
                pose_results = pose_model.predict(
                    crops,
                    conf=POSE_CONF,
                    imgsz=POSE_IMGSZ,
                    device=device,
                    half=half_ok,
                    verbose=False,
                )

                for pr, (tid, ox, oy, cw, ch) in zip(pose_results, meta):
                    head = None

                    if getattr(pr, "keypoints", None) is not None and pr.keypoints is not None:
                        try:
                            kxy_all = pr.keypoints.xy.cpu().numpy()
                        except Exception:
                            kxy_all = None
                        try:
                            kconf_all = pr.keypoints.conf.cpu().numpy()
                        except Exception:
                            kconf_all = None

                        if kxy_all is not None and len(kxy_all) > 0:
                            best_i = pick_best_pose_index(pr, cw, ch)
                            best_i = int(max(0, min(len(kxy_all) - 1, best_i)))

                            kxy = kxy_all[best_i]
                            kconf = None
                            if kconf_all is not None and len(kconf_all) > best_i:
                                kconf = kconf_all[best_i]

                            # ROI 坐标系下更紧的整头框
                            hx1, hy1, hx2, hy2 = head_box_full_from_pose(cw, ch, kxy, kconf)

                            # 映射回原图
                            hx1 += ox
                            hx2 += ox
                            hy1 += oy
                            hy2 += oy
                            hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                            head = np.array([hx1, hy1, hx2, hy2], dtype=np.float32)

                    if head is None:
                        pbox, _ = tracks[tid]
                        hx1, hy1, hx2, hy2 = head_box_fallback_full(*pbox.tolist())
                        hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                        head = np.array([hx1, hy1, hx2, hy2], dtype=np.float32)

                    # EMA 平滑（仅对有稳定 tid 的人）
                    if tid in head_state:
                        prev = head_state[tid]
                        if iou(prev, head) < IOU_RESET_THRESH:
                            head_state[tid] = head
                        else:
                            head_state[tid] = ema_update(prev, head, HEAD_EMA_ALPHA)
                    else:
                        head_state[tid] = head

                    last_pose_update[tid] = frame_idx

        # 3b) 无 tid：只计算本帧 head（不写入 head_state）
        head_noid = []  # [(pbox, score, head_box)]
        if need_pose_noid:
            crops = []
            meta = []  # (pbox, score, ox, oy, cw, ch)
            for (pbox, score) in need_pose_noid:
                crop, ox, oy = crop_with_pad(frame, pbox, CROP_PAD_RATIO)
                if crop is None or crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                if cw < 10 or ch < 10:
                    continue
                crops.append(crop)
                meta.append((pbox, score, ox, oy, cw, ch))

            if crops:
                pose_results = pose_model.predict(
                    crops,
                    conf=POSE_CONF,
                    imgsz=POSE_IMGSZ,
                    device=device,
                    half=half_ok,
                    verbose=False,
                )

                for pr, (pbox, score, ox, oy, cw, ch) in zip(pose_results, meta):
                    head = None
                    if getattr(pr, "keypoints", None) is not None and pr.keypoints is not None:
                        try:
                            kxy_all = pr.keypoints.xy.cpu().numpy()
                        except Exception:
                            kxy_all = None
                        try:
                            kconf_all = pr.keypoints.conf.cpu().numpy()
                        except Exception:
                            kconf_all = None

                        if kxy_all is not None and len(kxy_all) > 0:
                            best_i = pick_best_pose_index(pr, cw, ch)
                            best_i = int(max(0, min(len(kxy_all) - 1, best_i)))
                            kxy = kxy_all[best_i]
                            kconf = None
                            if kconf_all is not None and len(kconf_all) > best_i:
                                kconf = kconf_all[best_i]

                            hx1, hy1, hx2, hy2 = head_box_full_from_pose(cw, ch, kxy, kconf)
                            hx1 += ox
                            hx2 += ox
                            hy1 += oy
                            hy2 += oy
                            hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                            head = np.array([hx1, hy1, hx2, hy2], dtype=np.float32)

                    if head is None:
                        hx1, hy1, hx2, hy2 = head_box_fallback_full(*pbox.tolist())
                        hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                        head = np.array([hx1, hy1, hx2, hy2], dtype=np.float32)

                    head_noid.append((pbox, score, head))

        # 4) 绘制（不显示 id）
        # 4a) 有 tid
        if tracks:
            for tid, (pbox, score) in tracks.items():
                px1, py1, px2, py2 = pbox.tolist()

                # 人框
                ix1, iy1, ix2, iy2 = map(int, [px1, py1, px2, py2])
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"human {score:.2f}",
                    (ix1, max(15, iy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # 头框
                if tid in head_state:
                    hx1, hy1, hx2, hy2 = head_state[tid].tolist()
                else:
                    hx1, hy1, hx2, hy2 = head_box_fallback_full(px1, py1, px2, py2)

                hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                jx1, jy1, jx2, jy2 = map(int, [hx1, hy1, hx2, hy2])
                cv2.rectangle(frame, (jx1, jy1), (jx2, jy2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "head",
                    (jx1, max(15, jy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # 4b) 无 tid（本帧结果）
        if head_noid:
            for (pbox, score, head) in head_noid:
                px1, py1, px2, py2 = pbox.tolist()
                ix1, iy1, ix2, iy2 = map(int, [px1, py1, px2, py2])
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"human {score:.2f}",
                    (ix1, max(15, iy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                hx1, hy1, hx2, hy2 = head.tolist()
                hx1, hy1, hx2, hy2 = clamp_box_draw(hx1, hy1, hx2, hy2, W, H)
                jx1, jy1, jx2, jy2 = map(int, [hx1, hy1, hx2, hy2])
                cv2.rectangle(frame, (jx1, jy1), (jx2, jy2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "head",
                    (jx1, max(15, jy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # 5) 清理 stale tracks（仅 tid）
        if frame_idx % 10 == 0:
            stale = [tid for tid, t in last_seen.items() if (frame_idx - t) > STALE_FRAMES]
            for tid in stale:
                last_seen.pop(tid, None)
                head_state.pop(tid, None)
                last_pose_update.pop(tid, None)

        # 6) FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            inst = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst

        cv2.putText(
            frame,
            f"Frame:{W}x{H}  det_scale:{det_scale:.2f}  FPS:{fps:.1f}  PoseStride:{POSE_STRIDE}  Quit:'-'/'ESC'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("-") or key == 27:  # '-' or ESC
            break

        frame = None
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
