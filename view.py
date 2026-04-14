# view.py
# -*- coding: utf-8 -*-
"""
多路摄像头实时摔倒检测（YOLOv11 Pose + OpenCV）+ pushplus 微信推送
并集成：训练好的“摔倒 vs 躺下”事件分类器（joblib）

依赖：
pip install ultralytics opencv-python requests joblib scikit-learn
（建议）pip install lapx   # 用于跟踪关联更稳定（可选）

训练模型（joblib）说明：
- 训练脚本输出：fall_event_clf.joblib（或你自定义名字）
- 该分类器用于区分“摔倒（快倒下）”与“躺下（慢慢躺）”
- 运行时对每个 track_id 维护滑动窗口特征，并输出摔倒概率

运行：
python view.py
"""

import os
import time
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque

import cv2
import numpy as np
import requests

# Ultralytics YOLO
from ultralytics import YOLO

# UI
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ML classifier
import joblib


# ===========================
# pushplus 推送
# ===========================

def pushplus_send(token: str, title: str, content_html: str) -> bool:
    """pushplus 推送（失败返回 False）"""
    url = "http://www.pushplus.plus/send/"
    payload = {"token": token, "title": title, "content": content_html, "template": "html"}
    try:
        r = requests.post(url, json=payload, timeout=5)
        return (r.status_code == 200)
    except Exception:
        return False


# ===========================
# 数学/特征
# ===========================

def angle_to_vertical(dx: float, dy: float) -> float:
    """向量与竖直向下方向(0,1)夹角：0=竖直，90=水平"""
    norm = math.sqrt(dx * dx + dy * dy) + 1e-9
    cosv = dy / norm
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def safe_use_cv2_cuda_resize(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """若 OpenCV 支持 CUDA 则用 GPU resize，否则回退 CPU"""
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gm = cv2.cuda_GpuMat()
            gm.upload(frame)
            rgm = cv2.cuda.resize(gm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            return rgm.download()
    except Exception:
        pass
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def mosaic(frames: List[np.ndarray], target_w: int = 1600) -> np.ndarray:
    """多路画面拼接"""
    if not frames:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    n = len(frames)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    h0, w0 = frames[0].shape[:2]
    cell_w = max(320, target_w // cols)
    cell_h = int(cell_w * h0 / max(1, w0))

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        resized = safe_use_cv2_cuda_resize(f, cell_w, cell_h)
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        canvas[y1:y2, x1:x2] = resized
    return canvas


def calc_pose_metrics(kpts_xy: np.ndarray, box_xyxy: np.ndarray) -> Dict[str, float]:
    """
    从姿态关键点 + 框计算核心指标（和你训练脚本保持一致）
    返回：
      bh, wh_ratio, body_ang, head_hip_v_rel, cy
    """
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    wh_ratio = bw / bh

    # COCO 17点索引：0鼻子，5左肩，6右肩，11左髋，12右髋
    nose = kpts_xy[0]
    l_sh, r_sh = kpts_xy[5], kpts_xy[6]
    l_hip, r_hip = kpts_xy[11], kpts_xy[12]

    sh_center = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    dx = float(hip_center[0] - sh_center[0])
    dy = float(hip_center[1] - sh_center[1])
    body_ang = angle_to_vertical(dx, dy)

    head_to_hip_v = abs(float(nose[1] - hip_center[1]))
    head_hip_v_rel = head_to_hip_v / (bh + 1e-9)

    cy = (y1 + y2) / 2.0

    return {
        "bh": float(bh),
        "wh_ratio": float(wh_ratio),
        "body_ang": float(body_ang),
        "head_hip_v_rel": float(head_hip_v_rel),
        "cy": float(cy),
        "box_x1": float(x1),
        "box_y1": float(y1),
        "box_x2": float(x2),
        "box_y2": float(y2),
    }


def is_lying_like(m: Dict[str, float],
                  wh_th: float = 1.35,
                  ang_th: float = 58.0,
                  headhip_rel_th: float = 0.16,
                  score_th: int = 2) -> bool:
    """单帧躺姿判定（偏严格，减少误报）"""
    cond_box = m["wh_ratio"] > wh_th
    cond_ang = m["body_ang"] > ang_th
    cond_headhip = m["head_hip_v_rel"] < headhip_rel_th
    score = int(cond_box) + int(cond_ang) + int(cond_headhip)
    return score >= score_th


def pctl(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def build_clip_feature_from_window(window: List[Dict[str, float]], sample_fps: float) -> np.ndarray:
    """
    将“滑动窗口帧序列”转换为训练时同款的特征向量
    window: 每帧一个 metrics dict（含 body_ang, wh_ratio, head_hip_v_rel, cy, bh）
    """
    if len(window) < 8:
        return np.zeros((19,), dtype=np.float32)

    dt = 1.0 / max(1e-6, float(sample_fps))

    body_ang_seq = np.array([w["body_ang"] for w in window], dtype=np.float32)
    wh_ratio_seq = np.array([w["wh_ratio"] for w in window], dtype=np.float32)
    headhip_seq = np.array([w["head_hip_v_rel"] for w in window], dtype=np.float32)
    cy_seq = np.array([w["cy"] for w in window], dtype=np.float32)
    bh_seq = np.array([w["bh"] for w in window], dtype=np.float32)

    lying_seq = np.array([1.0 if w.get("lying_like", False) else 0.0 for w in window], dtype=np.float32)

    ang_speed = np.diff(body_ang_seq) / dt
    wh_speed = np.diff(wh_ratio_seq) / dt
    vy = np.diff(cy_seq) / dt
    vy_norm = vy / (bh_seq[1:] + 1e-6)

    idxs = np.where(lying_seq > 0.5)[0]
    time_to_lying = float(idxs[0] * dt) if len(idxs) > 0 else float(len(lying_seq) * dt)

    feats = [
        float(np.mean(body_ang_seq)),
        float(np.max(body_ang_seq)),
        pctl(body_ang_seq, 95),

        float(np.mean(wh_ratio_seq)),
        float(np.max(wh_ratio_seq)),
        pctl(wh_ratio_seq, 95),

        float(np.mean(headhip_seq)),
        float(np.min(headhip_seq)),

        float(np.mean(ang_speed)),
        float(np.max(ang_speed)),
        pctl(ang_speed, 95),

        float(np.mean(vy_norm)),
        float(np.max(vy_norm)),
        pctl(vy_norm, 95),

        float(np.mean(wh_speed)),
        float(np.max(wh_speed)),
        pctl(wh_speed, 95),

        float(np.mean(lying_seq)),
        float(time_to_lying),
    ]
    return np.array(feats, dtype=np.float32)



# ===========================
# 摄像头自动检测工具（Windows/OBS 兼容）
# ===========================

def open_camera_auto(prefer: int = 0, max_index: int = 10):
    """自动打开可用摄像头。
    - prefer：优先尝试的索引（比如 0）
    - max_index：最多扫描到哪个索引（不含）
    返回：(cap, picked_index)；失败返回 (None, -1)
    说明：
    - Windows 下优先尝试 DSHOW / MSMF 两种后端，常见于实体摄像头与 OBS 虚拟摄像头
    """
    # 先构造尝试顺序：优先 prefer，其余按 0..max_index-1
    order = [prefer] + [i for i in range(max_index) if i != prefer]

    backends = []
    # 有些 OpenCV 编译不包含某些后端，这里用 getattr 防御
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    # 最后兜底：默认后端
    backends.append(None)

    for idx in order:
        for be in backends:
            try:
                cap = cv2.VideoCapture(idx, be) if be is not None else cv2.VideoCapture(idx)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                return cap, idx
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    return None, -1


# ===========================
# 摄像头读取线程（自动重连）
# ===========================

class CameraReader(threading.Thread):
    """单路摄像头读取：只保留最新帧，尽量低延迟，支持断流重连"""
    def __init__(self,
                 cam_id: int,
                 source,
                 width: int,
                 height: int,
                 fps: int,
                 buffer_size: int,
                 name: str = ""):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.name = name or f"cam{cam_id}"

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0
        self.stop_flag = False

    def open(self) -> bool:
        """打开视频源
        - 如果 source 是数字（例如 0/1/2 或字符串 "0"），则按“摄像头索引”处理，并自动扫描可用设备
        - 否则按“视频文件/RTSP”等字符串源处理
        """
        # 1) 处理本地摄像头：允许 source 传入 int 或者 "0" 这样的数字字符串
        src = self.source
        cam_index: Optional[int] = None
        if isinstance(src, int):
            cam_index = int(src)
        elif isinstance(src, str) and src.strip().isdigit():
            cam_index = int(src.strip())

        if cam_index is not None:
            cap, picked = open_camera_auto(prefer=cam_index, max_index=10)
            if cap is None:
                return False
            if picked != cam_index:
                print(f"[信息] {self.name} 自动切换摄像头索引：{cam_index} -> {picked}")
            self.source = picked  # 记录为实际索引
            self.cap = cap
        else:
            # 2) RTSP/视频文件：尽量用 FFMPEG 后端（RTSP 更常见）
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            except Exception:
                self.cap = cv2.VideoCapture(self.source)

        # 通用参数设置
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass

        return bool(self.cap is not None and self.cap.isOpened())

    def run(self):
        retry_wait = 1.0
        max_wait = 10.0

        while not self.stop_flag:
            if self.cap is None or (not self.cap.isOpened()):
                ok = self.open()
                if not ok:
                    print(f"[警告] 打开失败，{self.name} 将在 {retry_wait:.1f}s 后重试：{self.source}")
                    time.sleep(retry_wait)
                    retry_wait = min(max_wait, retry_wait * 1.5)
                    continue
                else:
                    print(f"[信息] 打开成功：{self.name} -> {self.source}")
                    retry_wait = 1.0

            ok, frame = self.cap.read()
            if not ok or frame is None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(0.2)
                continue

            ts = time.time()
            with self.lock:
                self.latest_frame = frame
                self.latest_ts = ts

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_ts

    def stop(self):
        self.stop_flag = True


# ===========================
# 每个 track 的状态
# ===========================

@dataclass
class TrackState:
    # 躺姿连续帧
    consec_lying: int = 0

    # 推送冷却
    last_push_ts: float = 0.0

    # 最近出现时间（用于清理）
    last_seen_ts: float = 0.0

    # 规则法：最近一次“快速倒下事件”时间
    last_fall_event_ts: float = 0.0

    # 规则法：上一帧用于速度
    prev_ts: float = 0.0
    prev_body_ang: float = 0.0
    prev_cy: float = 0.0
    prev_wh_ratio: float = 0.0
    prev_is_standlike: bool = False

    # ML：滑动窗口（存 metrics）
    window: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=20))  # 默认约2秒(10fps)


# ===========================
# 视频处理主线程
# ===========================

class VideoProcessor(threading.Thread):
    """
    后台线程：多摄像头采集 -> YOLO pose 跟踪推理 -> 摔倒判定 -> 显示/推送
    """
    def __init__(self,
                 sources: List[str],
                 yolo_model_path: str,
                 algo_mode: str,
                 clf_path: str,
                 push_token: str,
                 infer_imgsz: int = 960,
                 conf_thres: float = 0.40,
                 use_half: bool = True,
                 capture_width: int = 1920,
                 capture_height: int = 1080,
                 capture_fps: int = 30,
                 buffer_size: int = 1,
                 # 误报优先：严格参数
                 fall_pose_consec_frames: int = 12,
                 fall_event_window_sec: float = 1.0,
                 push_cooldown_sec: int = 45,
                 ang_stand_max: float = 32.0,
                 ang_lying_min: float = 58.0,
                 wh_ratio_lying_th: float = 1.35,
                 head_hip_v_rel_th: float = 0.16,
                 lying_score_th: int = 2,
                 ang_speed_th: float = 190.0,
                 vy_norm_th: float = 1.45,
                 wh_speed_th: float = 1.80,
                 # ML 阈值（误报优先：阈值设高）
                 ml_prob_thres: float = 0.90,
                 ml_sample_fps: float = 10.0,
                 ):
        super().__init__(daemon=True)
        self.sources = sources
        self.yolo_model_path = yolo_model_path
        self.algo_mode = algo_mode
        self.clf_path = clf_path
        self.push_token = push_token

        self.infer_imgsz = infer_imgsz
        self.conf_thres = conf_thres
        self.use_half = use_half

        self.capture_width = capture_width
        self.capture_height = capture_height
        self.capture_fps = capture_fps
        self.buffer_size = buffer_size

        self.fall_pose_consec_frames = fall_pose_consec_frames
        self.fall_event_window_sec = fall_event_window_sec
        self.push_cooldown_sec = push_cooldown_sec

        self.ang_stand_max = ang_stand_max
        self.ang_lying_min = ang_lying_min
        self.wh_ratio_lying_th = wh_ratio_lying_th
        self.head_hip_v_rel_th = head_hip_v_rel_th
        self.lying_score_th = lying_score_th

        self.ang_speed_th = ang_speed_th
        self.vy_norm_th = vy_norm_th
        self.wh_speed_th = wh_speed_th

        self.ml_prob_thres = ml_prob_thres
        self.ml_sample_fps = ml_sample_fps

        self.stop_flag = False
        self.readers: List[CameraReader] = []
        self.states_per_cam: List[Dict[int, TrackState]] = []

        self.model: Optional[YOLO] = None
        self.clf = None  # joblib classifier
        self.device = None

        self.last_canvas = None
        self.last_status_text = ""

    def stop(self):
        self.stop_flag = True

    def _check_cuda_or_raise(self):
        """你说必须 CUDA：这里强制检查"""
        import torch
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise RuntimeError(
                "CUDA 不可用：请确认你安装的是 cu128 版 torch，且 nvidia 驱动正常。"
            )
        self.device = "cuda:0"

    def _load_models(self):
        self._check_cuda_or_raise()

        # 加载 YOLO
        print(f"[信息] 加载 YOLO 模型：{self.yolo_model_path}")
        self.model = YOLO(self.yolo_model_path)

        # half 仅 GPU
        self.use_half = bool(self.use_half)

        # 加载分类器（可选）
        if self.algo_mode in ("训练模型", "混合") and self.clf_path and os.path.isfile(self.clf_path):
            print(f"[信息] 加载训练分类器：{self.clf_path}")
            self.clf = joblib.load(self.clf_path)
        else:
            self.clf = None
            if self.algo_mode in ("训练模型", "混合"):
                print("[警告] 未加载训练分类器（joblib不存在或未选择），将无法使用训练模型模式。")

    def _start_readers(self):
        self.readers = []
        for i, src in enumerate(self.sources):
            rd = CameraReader(
                cam_id=i,
                source=src,
                width=self.capture_width,
                height=self.capture_height,
                fps=self.capture_fps,
                buffer_size=self.buffer_size,
                name=f"cam{i}"
            )
            rd.start()
            self.readers.append(rd)
        self.states_per_cam = [dict() for _ in self.readers]

    def _stop_readers(self):
        for rd in self.readers:
            rd.stop()
        self.readers = []

    def _rule_fast_event(self, st: TrackState, m: Dict[str, float], frame_ts: float) -> Tuple[bool, Dict[str, float]]:
        """
        规则法快速倒下事件：角速度 / 归一化下落速度 / 宽高比变化速度
        返回 fast_event 及调试值
        """
        body_ang = m["body_ang"]
        wh_ratio = m["wh_ratio"]
        bh = max(1.0, m["bh"])
        cy = m["cy"]

        dt = 0.0
        ang_speed = 0.0
        vy_norm = 0.0
        wh_speed = 0.0

        if st.prev_ts > 0.0:
            dt = max(1e-3, frame_ts - st.prev_ts)
            ang_speed = (body_ang - st.prev_body_ang) / dt

            vy = (cy - st.prev_cy) / dt
            vy_norm = vy / bh

            wh_speed = (wh_ratio - st.prev_wh_ratio) / dt

        is_standlike = body_ang < self.ang_stand_max

        fast_event = False
        if dt > 0.0:
            if ang_speed > self.ang_speed_th:
                fast_event = True
            if vy_norm > self.vy_norm_th:
                fast_event = True
            if wh_speed > self.wh_speed_th:
                fast_event = True

            # 竖直到水平的跃迁（再加一道更严格门槛）
            if st.prev_is_standlike and (body_ang > self.ang_lying_min) and (dt < 0.7):
                if (ang_speed > (0.8 * self.ang_speed_th)) or (wh_speed > (0.8 * self.wh_speed_th)):
                    fast_event = True

        # 更新上一帧
        st.prev_ts = frame_ts
        st.prev_body_ang = body_ang
        st.prev_cy = cy
        st.prev_wh_ratio = wh_ratio
        st.prev_is_standlike = is_standlike

        dbg = {
            "dt": dt,
            "ang_speed": ang_speed,
            "vy_norm": vy_norm,
            "wh_speed": wh_speed,
        }
        return fast_event, dbg

    def _ml_event_prob(self, st: TrackState) -> float:
        """
        训练模型输出“摔倒”概率（越大越像摔倒事件）
        """
        if self.clf is None:
            return 0.0
        window_list = list(st.window)
        feats = build_clip_feature_from_window(window_list, self.ml_sample_fps).reshape(1, -1)
        try:
            proba = float(self.clf.predict_proba(feats)[0, 1])
            return proba
        except Exception:
            # 有些模型可能没有 predict_proba
            try:
                pred = int(self.clf.predict(feats)[0])
                return 1.0 if pred == 1 else 0.0
            except Exception:
                return 0.0

    def run(self):
        try:
            self._load_models()
        except Exception as e:
            self.last_status_text = f"启动失败：{e}"
            print("[错误]", self.last_status_text)
            return

        self._start_readers()

        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

        last_fps_ts = time.time()
        loop_frames = 0
        loop_fps = 0.0

        try:
            while not self.stop_flag:
                # 批量取帧
                batch_frames = []
                batch_meta = []  # (cam_i, frame_ts)
                for cam_i, rd in enumerate(self.readers):
                    frame, ts = rd.get_latest()
                    if frame is None:
                        continue
                    batch_frames.append(frame)
                    batch_meta.append((cam_i, ts))

                if not batch_frames:
                    time.sleep(0.01)
                    continue

                # 批量推理（track 保持 ID）
                results = self.model.track(
                    source=batch_frames,
                    stream=False,
                    device=self.device,
                    half=self.use_half,
                    imgsz=self.infer_imgsz,
                    conf=self.conf_thres,
                    verbose=False,
                    persist=True,
                )

                now = time.time()
                annotated_frames = []

                for idx, res in enumerate(results):
                    cam_i, frame_ts = batch_meta[idx]
                    frame = batch_frames[idx]
                    h, w = frame.shape[:2]

                    # 清理过期 track
                    stmap = self.states_per_cam[cam_i]
                    stale = [tid for tid, st in stmap.items() if (now - st.last_seen_ts) > 5.0]
                    for tid in stale:
                        stmap.pop(tid, None)

                    boxes = getattr(res, "boxes", None)
                    kpts = getattr(res, "keypoints", None)

                    if boxes is None or kpts is None or len(boxes) == 0:
                        cv2.putText(frame, f"CAM{cam_i} no person", (15, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        annotated_frames.append(frame)
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)

                    # 跟踪ID（无则临时）
                    ids = None
                    try:
                        if boxes.id is not None:
                            ids = boxes.id.cpu().numpy().astype(int)
                    except Exception:
                        ids = None

                    kpts_xy_all = kpts.xy.cpu().numpy()  # (N,17,2)

                    for j in range(len(xyxy)):
                        tid = int(ids[j]) if (ids is not None and j < len(ids)) else int(100000 + j)

                        st = stmap.get(tid, TrackState(window=deque(maxlen=int(self.ml_sample_fps * 2))))
                        st.last_seen_ts = now

                        # 计算 metrics
                        m = calc_pose_metrics(kpts_xy_all[j], xyxy[j])

                        # 躺姿（单帧）
                        m["lying_like"] = is_lying_like(
                            m,
                            wh_th=self.wh_ratio_lying_th,
                            ang_th=self.ang_lying_min,
                            headhip_rel_th=self.head_hip_v_rel_th,
                            score_th=self.lying_score_th
                        )

                        # 更新滑动窗口（用于 ML）
                        # 为了近似 SAMPLE_FPS，我们按“时间间隔”抽样存入 window
                        if len(st.window) == 0:
                            st.window.append(m)
                        else:
                            # 控制窗口采样率：如果两次存入间隔 < 1/SAMPLE_FPS，就不存（降低重复）
                            last_t = st.window[-1].get("_ts", 0.0)
                            if last_t == 0.0 or (frame_ts - last_t) >= (1.0 / self.ml_sample_fps):
                                m["_ts"] = frame_ts
                                st.window.append(m)

                        # 躺姿连续帧计数（严格：回退更强）
                        if m["lying_like"]:
                            st.consec_lying += 1
                        else:
                            st.consec_lying = max(0, st.consec_lying - 2)

                        # 三种模式：规则 / 训练模型 / 混合
                        rule_fast, dbg_rule = self._rule_fast_event(st, m, frame_ts)
                        if rule_fast:
                            st.last_fall_event_ts = now

                        in_rule_event_window = (now - st.last_fall_event_ts) <= self.fall_event_window_sec
                        rule_is_fall = in_rule_event_window and (st.consec_lying >= self.fall_pose_consec_frames)

                        ml_prob = self._ml_event_prob(st)
                        ml_is_fall = (ml_prob >= self.ml_prob_thres) and (st.consec_lying >= self.fall_pose_consec_frames)

                        if self.algo_mode == "规则":
                            is_fall = rule_is_fall
                            algo_tag = "RULE"
                        elif self.algo_mode == "训练模型":
                            # 训练模型模式：只信 ML（误报优先：阈值很高）
                            is_fall = ml_is_fall
                            algo_tag = "ML"
                        else:
                            # 混合模式（最稳）：规则快速事件 + ML 概率双重通过
                            is_fall = rule_is_fall and (ml_prob >= (self.ml_prob_thres * 0.9))
                            algo_tag = "HYB"

                        # 画框
                        x1, y1, x2, y2 = int(m["box_x1"]), int(m["box_y1"]), int(m["box_x2"]), int(m["box_y2"])
                        if is_fall:
                            color = (0, 0, 255)
                        elif m["lying_like"]:
                            color = (0, 200, 255)
                        else:
                            color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # 叠加调试文字（方便你现场调参）
                        txt1 = f"{algo_tag} CAM{cam_i} ID:{tid} conf:{confs[j]:.2f} ang:{m['body_ang']:.1f} lieN:{st.consec_lying}"
                        txt2 = f"angSp:{dbg_rule['ang_speed']:.0f} vyN:{dbg_rule['vy_norm']:.2f} whSp:{dbg_rule['wh_speed']:.2f} mlP:{ml_prob:.2f}"
                        cv2.putText(frame, txt1, (x1, max(0, y1 - 26)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, txt2, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 画关键点（点）
                        pts = kpts_xy_all[j].astype(int)
                        for p in pts:
                            cv2.circle(frame, (p[0], p[1]), 2, (255, 255, 0), -1)

                        # 推送（冷却）
                        if is_fall and ((now - st.last_push_ts) >= self.push_cooldown_sec):
                            st.last_push_ts = now
                            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                            content = f"""
                            <b>检测到疑似摔倒（{algo_tag}，误报优先严格）！</b><br>
                            摄像头：CAM{cam_i}<br>
                            时间：{tstr}<br>
                            目标ID：{tid}<br>
                            躺姿连续帧：{st.consec_lying}<br>
                            角度：{m['body_ang']:.1f}°<br>
                            角速度：{dbg_rule['ang_speed']:.0f}°/s<br>
                            归一化下落速度：{dbg_rule['vy_norm']:.2f} (框高)/s<br>
                            宽高比变化速度：{dbg_rule['wh_speed']:.2f} /s<br>
                            训练模型摔倒概率：{ml_prob:.2f}<br>
                            """

                            if self.push_token and ("请在这里填写" not in self.push_token):
                                ok = pushplus_send(self.push_token, "⚠️ 摔倒告警", content)
                                print(f"[推送] CAM{cam_i} ID{tid} -> {'成功' if ok else '失败'}")
                            else:
                                print(f"[告警模拟] CAM{cam_i} ID{tid} {tstr}")

                        stmap[tid] = st

                    cv2.putText(frame, f"CAM{cam_i} {w}x{h}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    annotated_frames.append(frame)

                # FPS
                loop_frames += 1
                if time.time() - last_fps_ts >= 1.0:
                    loop_fps = loop_frames / (time.time() - last_fps_ts)
                    loop_frames = 0
                    last_fps_ts = time.time()

                canvas = mosaic(annotated_frames, target_w=1600)
                info = f"FPS:{loop_fps:.1f} imgsz:{self.infer_imgsz} device:{self.device} algo:{self.algo_mode} mlTH:{self.ml_prob_thres:.2f}"
                cv2.putText(canvas, info, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                self.last_canvas = canvas
                self.last_status_text = info

                cv2.imshow("Video", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    self.stop_flag = True
                    break

        finally:
            self._stop_readers()
            try:
                cv2.destroyWindow("Video")
            except Exception:
                pass
            print("[信息] VideoProcessor 已停止。")


# ===========================
# Tkinter UI
# ===========================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("摔倒检测控制面板（YOLOv11 Pose + 训练模型）")
        self.processor: Optional[VideoProcessor] = None

        # 默认参数（你可以改）
        self.var_push_token = tk.StringVar(value="请在这里填写你的pushplus_token")
        self.var_yolo_model = tk.StringVar(value="yolo11n-pose.pt")
        self.var_clf_path = tk.StringVar(value="fall_event_clf.joblib")

        self.var_algo = tk.StringVar(value="混合")  # 规则/训练模型/混合

        self.var_imgsz = tk.IntVar(value=960)
        self.var_conf = tk.DoubleVar(value=0.40)
        self.var_half = tk.BooleanVar(value=True)

        self.var_pose_consec = tk.IntVar(value=12)
        self.var_event_window = tk.DoubleVar(value=1.0)
        self.var_push_cd = tk.IntVar(value=45)

        self.var_ml_prob = tk.DoubleVar(value=0.90)
        self.var_ml_fps = tk.DoubleVar(value=10.0)

        self.var_cap_w = tk.IntVar(value=1920)
        self.var_cap_h = tk.IntVar(value=1080)
        self.var_cap_fps = tk.IntVar(value=30)
        self.var_buf = tk.IntVar(value=1)

        self.start_time = None
        self.timer_job = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, **pad)

        # 摄像头源
        ttk.Label(frm, text="摄像头源（每行一个：本地用 0/1，RTSP 用 rtsp://... ）").grid(row=0, column=0, columnspan=3, sticky="w")
        self.txt_sources = tk.Text(frm, width=70, height=6)
        self.txt_sources.grid(row=1, column=0, columnspan=3, sticky="we")
        self.txt_sources.insert("1.0", "0\n")

        # 模型选择
        ttk.Label(frm, text="YOLO Pose 模型：").grid(row=2, column=0, sticky="e")
        yolo_entry = ttk.Entry(frm, textvariable=self.var_yolo_model, width=45)
        yolo_entry.grid(row=2, column=1, sticky="we")
        ttk.Button(frm, text="选择文件", command=self._pick_yolo).grid(row=2, column=2, sticky="we")

        ttk.Label(frm, text="训练模型 joblib：").grid(row=3, column=0, sticky="e")
        clf_entry = ttk.Entry(frm, textvariable=self.var_clf_path, width=45)
        clf_entry.grid(row=3, column=1, sticky="we")
        ttk.Button(frm, text="选择文件", command=self._pick_clf).grid(row=3, column=2, sticky="we")

        # 算法模式
        ttk.Label(frm, text="算法模式：").grid(row=4, column=0, sticky="e")
        algo_combo = ttk.Combobox(frm, textvariable=self.var_algo, values=["规则", "训练模型", "混合"], state="readonly")
        algo_combo.grid(row=4, column=1, sticky="w")

        # pushplus
        ttk.Label(frm, text="pushplus token：").grid(row=5, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_push_token, width=45).grid(row=5, column=1, sticky="we")
        ttk.Button(frm, text="测试推送", command=self._test_push).grid(row=5, column=2, sticky="we")

        # 推理参数
        sep1 = ttk.Separator(frm, orient="horizontal")
        sep1.grid(row=6, column=0, columnspan=3, sticky="we", pady=8)

        ttk.Label(frm, text="imgsz：").grid(row=7, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_imgsz, width=10).grid(row=7, column=1, sticky="w")
        ttk.Label(frm, text="conf：").grid(row=7, column=1, sticky="e", padx=(140, 0))
        ttk.Entry(frm, textvariable=self.var_conf, width=10).grid(row=7, column=2, sticky="w")
        ttk.Checkbutton(frm, text="half(GPU)", variable=self.var_half).grid(row=8, column=1, sticky="w")

        # 误报优先参数（关键）
        ttk.Label(frm, text="躺姿连续帧：").grid(row=9, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_pose_consec, width=10).grid(row=9, column=1, sticky="w")
        ttk.Label(frm, text="事件窗口(s)：").grid(row=9, column=1, sticky="e", padx=(140, 0))
        ttk.Entry(frm, textvariable=self.var_event_window, width=10).grid(row=9, column=2, sticky="w")

        ttk.Label(frm, text="推送冷却(s)：").grid(row=10, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_push_cd, width=10).grid(row=10, column=1, sticky="w")

        # ML 参数
        ttk.Label(frm, text="ML摔倒概率阈值：").grid(row=11, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_ml_prob, width=10).grid(row=11, column=1, sticky="w")
        ttk.Label(frm, text="ML窗口采样fps：").grid(row=11, column=1, sticky="e", padx=(140, 0))
        ttk.Entry(frm, textvariable=self.var_ml_fps, width=10).grid(row=11, column=2, sticky="w")

        # 采集参数
        sep2 = ttk.Separator(frm, orient="horizontal")
        sep2.grid(row=12, column=0, columnspan=3, sticky="we", pady=8)

        ttk.Label(frm, text="采集宽：").grid(row=13, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_cap_w, width=10).grid(row=13, column=1, sticky="w")
        ttk.Label(frm, text="采集高：").grid(row=13, column=1, sticky="e", padx=(140, 0))
        ttk.Entry(frm, textvariable=self.var_cap_h, width=10).grid(row=13, column=2, sticky="w")

        ttk.Label(frm, text="采集FPS：").grid(row=14, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_cap_fps, width=10).grid(row=14, column=1, sticky="w")
        ttk.Label(frm, text="缓冲：").grid(row=14, column=1, sticky="e", padx=(140, 0))
        ttk.Entry(frm, textvariable=self.var_buf, width=10).grid(row=14, column=2, sticky="w")

        # 控制按钮 + 计时器
        sep3 = ttk.Separator(frm, orient="horizontal")
        sep3.grid(row=15, column=0, columnspan=3, sticky="we", pady=8)

        self.btn_start = ttk.Button(frm, text="开始", command=self.start)
        self.btn_start.grid(row=16, column=0, sticky="we")

        self.btn_stop = ttk.Button(frm, text="结束", command=self.stop, state="disabled")
        self.btn_stop.grid(row=16, column=1, sticky="we")

        self.lbl_timer = ttk.Label(frm, text="计时：00:00:00")
        self.lbl_timer.grid(row=16, column=2, sticky="e")

        self.lbl_status = ttk.Label(frm, text="状态：未运行")
        self.lbl_status.grid(row=17, column=0, columnspan=3, sticky="w")

        frm.columnconfigure(1, weight=1)

    def _pick_yolo(self):
        fp = filedialog.askopenfilename(title="选择 YOLO Pose 模型文件", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if fp:
            self.var_yolo_model.set(fp)

    def _pick_clf(self):
        fp = filedialog.askopenfilename(title="选择 训练模型 joblib 文件", filetypes=[("Joblib", "*.joblib"), ("All", "*.*")])
        if fp:
            self.var_clf_path.set(fp)

    def _test_push(self):
        token = self.var_push_token.get().strip()
        if (not token) or ("请在这里填写" in token):
            messagebox.showwarning("提示", "请先填写 pushplus token。")
            return
        ok = pushplus_send(token, "✅ pushplus 测试", "<b>测试推送成功</b>")
        messagebox.showinfo("结果", "推送成功" if ok else "推送失败（请检查 token / 网络）")

    def _parse_sources(self) -> List[str]:
        raw = self.txt_sources.get("1.0", "end").strip().splitlines()
        sources = []
        for line in raw:
            s = line.strip()
            if not s:
                continue
            # 支持写 0/1 这类本地摄像头
            if s.isdigit():
                sources.append(int(s))
            else:
                sources.append(s)
        return sources

    def start(self):
        if self.processor is not None and self.processor.is_alive():
            messagebox.showinfo("提示", "已经在运行中。")
            return

        # 强制 CUDA：提前给用户明确提示
        try:
            import torch
            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
                messagebox.showerror("CUDA 不可用",
                                     "检测不到 CUDA。\n\n"
                                     "请确认：\n"
                                     "1) 已安装 cu128 版 torch（不是 +cpu）\n"
                                     "2) nvidia-smi 能正常显示显卡\n"
                                     "3) 你的虚拟环境里没有混装多个 torch\n")
                return
        except Exception as e:
            messagebox.showerror("错误", f"无法检查 CUDA：{e}")
            return

        sources = self._parse_sources()
        if not sources:
            messagebox.showwarning("提示", "请至少填写一个摄像头源。")
            return

        yolo_path = self.var_yolo_model.get().strip()
        if not yolo_path:
            messagebox.showwarning("提示", "请填写 YOLO 模型路径。")
            return

        algo_mode = self.var_algo.get().strip()
        clf_path = self.var_clf_path.get().strip()

        # 训练模型模式必须有 joblib
        if algo_mode in ("训练模型", "混合"):
            if (not clf_path) or (not os.path.isfile(clf_path)):
                messagebox.showwarning("提示", "你选择了训练模型/混合模式，但 joblib 文件不存在。\n请先选择训练模型文件。")
                return

        self.processor = VideoProcessor(
            sources=sources,
            yolo_model_path=yolo_path,
            algo_mode=algo_mode,
            clf_path=clf_path,
            push_token=self.var_push_token.get().strip(),

            infer_imgsz=int(self.var_imgsz.get()),
            conf_thres=float(self.var_conf.get()),
            use_half=bool(self.var_half.get()),

            capture_width=int(self.var_cap_w.get()),
            capture_height=int(self.var_cap_h.get()),
            capture_fps=int(self.var_cap_fps.get()),
            buffer_size=int(self.var_buf.get()),

            fall_pose_consec_frames=int(self.var_pose_consec.get()),
            fall_event_window_sec=float(self.var_event_window.get()),
            push_cooldown_sec=int(self.var_push_cd.get()),

            ml_prob_thres=float(self.var_ml_prob.get()),
            ml_sample_fps=float(self.var_ml_fps.get()),
        )

        self.processor.start()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="状态：运行中（视频窗口已打开，按 q 或 ESC 也可退出）")

        self.start_time = time.time()
        self._tick_timer()
        self._poll_status()

    def stop(self):
        if self.processor is not None:
            self.processor.stop()
        self.processor = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="状态：已停止")

        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.lbl_timer.config(text="计时：00:00:00")

    def _tick_timer(self):
        if self.start_time is None:
            return
        elapsed = int(time.time() - self.start_time)
        hh = elapsed // 3600
        mm = (elapsed % 3600) // 60
        ss = elapsed % 60
        self.lbl_timer.config(text=f"计时：{hh:02d}:{mm:02d}:{ss:02d}")
        self.timer_job = self.root.after(500, self._tick_timer)

    def _poll_status(self):
        # 定时刷新状态文本
        if self.processor is not None:
            if (not self.processor.is_alive()):
                # 线程自己退出了（例如按了 q 或异常）
                msg = self.processor.last_status_text or "已退出"
                self.stop()
                messagebox.showinfo("提示", f"检测线程已停止：{msg}")
                return
            else:
                if self.processor.last_status_text:
                    self.lbl_status.config(text=f"状态：{self.processor.last_status_text}")
        self.root.after(500, self._poll_status)


def main():
    # 尽量避免 Windows 控制台编码问题
    os.environ.setdefault("PYTHONUTF8", "1")

    root = tk.Tk()
    app = App(root)

    def on_close():
        try:
            app.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
