# -*- coding: utf-8 -*-
"""
YOLOv11 单类（摔倒 fall）检测训练脚本
依赖：
pip install ultralytics opencv-python pyyaml numpy
# 事件分类器需要：
pip install scikit-learn joblib

运行流程：
- 第一次：python train.py  -> 选 make_dataset（抽帧）
- 第二次：python train.py  -> 选 auto_label（自动生成 labels）
- 第三次：python train.py  -> 选 train（正式训练 best.pt）
- 第四次：python train.py  -> 选 train_event_clf（生成 fall_event_clf.joblib）

训练输出（检测模型）：
./模型/exp/weights/best.pt

事件分类输出：
./fall_event_clf.joblib
"""

import os
import sys
import glob
import random
import shutil
import argparse
from typing import List, Tuple, Dict, Optional

import cv2
import yaml
import numpy as np
from ultralytics import YOLO


# =========================
# 路径默认值（同目录 摔倒/ 与 模型/）
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FALL_RAW = os.path.join(BASE_DIR, "摔倒")        # ./摔倒（你当前就是这个）
DEFAULT_PROJECT_DIR = os.path.join(BASE_DIR, "模型")      # ./模型
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # ./dataset
DEFAULT_POSE_MODEL = os.path.join(BASE_DIR, "yolo11n-pose.pt")  # 默认 pose 模型（可改）

# ✅新增：事件分类器输出
DEFAULT_JOBLIB_OUT = os.path.join(BASE_DIR, "fall_event_clf.joblib")


# ---------------------------
# 对话式输入工具
# ---------------------------

def ask_str(prompt: str, default: str) -> str:
    s = input(f"{prompt}（默认：{default}）: ").strip()
    return s if s else default

def ask_int(prompt: str, default: int, minv: int = None, maxv: int = None) -> int:
    while True:
        s = input(f"{prompt}（默认：{default}）: ").strip()
        if not s:
            v = default
        else:
            try:
                v = int(s)
            except Exception:
                print("输入无效，请输入整数。")
                continue
        if minv is not None and v < minv:
            print(f"不能小于 {minv}")
            continue
        if maxv is not None and v > maxv:
            print(f"不能大于 {maxv}")
            continue
        return v

def ask_float(prompt: str, default: float, minv: float = None, maxv: float = None) -> float:
    while True:
        s = input(f"{prompt}（默认：{default}）: ").strip()
        if not s:
            v = default
        else:
            try:
                v = float(s)
            except Exception:
                print("输入无效，请输入数字。")
                continue
        if minv is not None and v < minv:
            print(f"不能小于 {minv}")
            continue
        if maxv is not None and v > maxv:
            print(f"不能大于 {maxv}")
            continue
        return v

def ask_choice(prompt: str, choices: List[str], default: str) -> str:
    cset = set(choices)
    while True:
        s = input(f"{prompt} {choices}（默认：{default}）: ").strip()
        if not s:
            return default
        if s in cset:
            return s
        print("输入无效，请从列表中选择。")


# ---------------------------
# 通用工具函数
# ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_media_files(folder: str) -> List[str]:
    exts = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.m4v", "*.webm",
            "*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def is_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"]

def write_yaml(path: str, dataset_dir: str):
    """单类数据集 YAML：只包含 fall 一个类（class=0）"""
    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "fall"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def split_train_val(items: List[str], val_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 2 else 0
    val = items[:n_val]
    train = items[n_val:]
    return train, val

def extract_frames_from_video(video_path: str, out_dir: str, extract_fps: float) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[警告] 无法打开视频：{video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0
    step = max(1, int(round(fps / max(0.1, extract_fps))))

    base = os.path.splitext(os.path.basename(video_path))[0]
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % step == 0:
            out_name = f"{base}_{idx:06d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved += 1
        idx += 1

    cap.release()
    return saved

def copy_or_extract(raw_dir: str, out_dir: str, extract_fps: float):
    """raw_dir 可放视频或图片：视频抽帧、图片直接复制"""
    ensure_dir(out_dir)
    files = list_media_files(raw_dir)
    if not files:
        raise RuntimeError(f"素材目录为空：{raw_dir}")

    total = 0
    for fp in files:
        if is_video(fp):
            n = extract_frames_from_video(fp, out_dir, extract_fps)
            print(f"[抽帧] {os.path.basename(fp)} -> {n} 帧")
            total += n
        else:
            dst = os.path.join(out_dir, os.path.basename(fp))
            shutil.copy2(fp, dst)
            total += 1
    print(f"[完成] 共生成/复制 {total} 张图片 -> {out_dir}")

def check_labels_exist(dataset_dir: str) -> Tuple[int, int]:
    lt = glob.glob(os.path.join(dataset_dir, "labels", "train", "*.txt"))
    lv = glob.glob(os.path.join(dataset_dir, "labels", "val", "*.txt"))
    return len(lt), len(lv)

def check_cuda_or_exit():
    """必须 CUDA：不满足直接退出"""
    import torch
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        print("【错误】CUDA 不可用：你的 torch 可能是 CPU 版（+cpu），或驱动未正确安装。")
        print("请先确保安装 cu128 版 torch，并能通过 nvidia-smi 查看显卡。")
        sys.exit(1)
    print(f"[信息] CUDA 可用：GPU数量={torch.cuda.device_count()}，GPU0={torch.cuda.get_device_name(0)}")


# ---------------------------
# auto_label：躺姿判定 + 写 YOLO label
# ---------------------------

def angle_to_vertical(dx: float, dy: float) -> float:
    """向量与竖直向下(0,1)夹角：0=竖直，90=水平"""
    norm = (dx * dx + dy * dy) ** 0.5 + 1e-9
    cosv = dy / norm
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))

def calc_pose_metrics(kpts_xy: np.ndarray, box_xyxy: np.ndarray) -> Dict[str, float]:
    """
    从姿态关键点 + 框计算核心指标（与 view.py 的 calc_pose_metrics 对齐）
    返回：bh, wh_ratio, body_ang, head_hip_v_rel, cy 等
    """
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    wh_ratio = bw / bh

    # COCO17: 0鼻子，5左肩，6右肩，11左髋，12右髋
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
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
    }

def is_lying_like(m: Dict[str, float],
                  wh_th: float,
                  ang_th: float,
                  headhip_rel_th: float,
                  score_th: int) -> bool:
    """
    偏严格的单帧躺姿判定（减少误标/误报）
    满足 >= score_th 个条件视为“像躺姿”
    """
    cond1 = m["wh_ratio"] > wh_th
    cond2 = m["body_ang"] > ang_th
    cond3 = m["head_hip_v_rel"] < headhip_rel_th
    score = int(cond1) + int(cond2) + int(cond3)
    return score >= score_th

def xyxy_to_yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """把像素框转 YOLO归一化格式：class xc yc bw bh（都0~1）"""
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0

    xc /= w
    yc /= h
    bw /= w
    bh /= h

    # 单类 fall => class=0
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

def auto_label_folder(images_dir: str,
                      labels_dir: str,
                      pose_model_path: str,
                      imgsz: int,
                      conf: float,
                      wh_th: float,
                      ang_th: float,
                      headhip_rel_th: float,
                      score_th: int,
                      overwrite: bool,
                      save_debug: bool):
    """
    对 images_dir 下图片自动标注，写到 labels_dir
    - 只标注“躺姿判定通过”的帧（减少误标）
    - 每张图只取“最大人框”作为摔倒对象（简化）
    """
    ensure_dir(labels_dir)
    img_files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_files.extend(glob.glob(os.path.join(images_dir, e)))
    img_files = sorted(img_files)
    if not img_files:
        print(f"[警告] 目录没有图片：{images_dir}")
        return 0, 0

    check_cuda_or_exit()
    model = YOLO(pose_model_path)

    labeled = 0
    skipped = 0

    dbg_dir = os.path.join(os.path.dirname(images_dir), "_debug_autolabel")
    if save_debug:
        ensure_dir(dbg_dir)

    for i, fp in enumerate(img_files, 1):
        img = cv2.imread(fp)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # pose 推理
        res = model.predict(img, imgsz=imgsz, conf=conf, device=0, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        kpts = getattr(res, "keypoints", None)
        if boxes is None or kpts is None or len(boxes) == 0:
            skipped += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        kxy = kpts.xy.cpu().numpy()

        # 取最大框的人（主角）
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        j = int(np.argmax(areas))

        m = calc_pose_metrics(kxy[j], xyxy[j])
        lying = is_lying_like(m, wh_th=wh_th, ang_th=ang_th, headhip_rel_th=headhip_rel_th, score_th=score_th)

        # label 路径
        stem = os.path.splitext(os.path.basename(fp))[0]
        out_txt = os.path.join(labels_dir, stem + ".txt")

        if not lying:
            # 不像倒地：默认不写 label（留空=背景），避免误标
            if overwrite and os.path.isfile(out_txt):
                os.remove(out_txt)
            skipped += 1
        else:
            # 写入摔倒框（class=0）
            line = xyxy_to_yolo_line(m["x1"], m["y1"], m["x2"], m["y2"], w, h)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(line)
            labeled += 1

            if save_debug:
                dbg = img.copy()
                x1, y1, x2, y2 = int(m["x1"]), int(m["y1"]), int(m["x2"]), int(m["y2"])
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                txt = f"lying=1 ang={m['body_ang']:.1f} wh={m['wh_ratio']:.2f} hh={m['head_hip_v_rel']:.2f}"
                cv2.putText(dbg, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(dbg_dir, stem + ".jpg"), dbg)

        if i % 50 == 0:
            print(f"[auto_label] {images_dir}: {i}/{len(img_files)}  已标注={labeled}  跳过={skipped}")

    print(f"[完成] {images_dir} 自动标注：已标注 {labeled} 张，跳过 {skipped} 张")
    return labeled, skipped


# ---------------------------
# 模式：make_dataset
# ---------------------------

def make_dataset(dataset_dir: str, fall_raw: str, extract_fps: float, val_ratio: float):
    """
    从 ./摔倒 抽帧/复制 -> ./dataset
    生成：
      dataset/images/train
      dataset/images/val
      dataset/labels/train（空）
      dataset/labels/val（空）
      dataset/data.yaml
    """
    if not os.path.isdir(fall_raw):
        raise RuntimeError(f"找不到摔倒素材目录：{fall_raw}\n请在 train.py 同目录下创建“摔倒”文件夹并放入视频/图片。")

    ensure_dir(dataset_dir)
    ensure_dir(os.path.join(dataset_dir, "images"))
    ensure_dir(os.path.join(dataset_dir, "labels"))

    all_fall_dir = os.path.join(dataset_dir, "images", "all_fall")
    copy_or_extract(fall_raw, all_fall_dir, extract_fps)

    imgs = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(glob.glob(os.path.join(all_fall_dir, e)))
    imgs = sorted(imgs)

    if len(imgs) < 30:
        print("[警告] 图片数量偏少，建议至少 100+（越多越好），否则精度与泛化会不稳定。")

    train_imgs, val_imgs = split_train_val(imgs, val_ratio, seed=42)

    train_dir = os.path.join(dataset_dir, "images", "train")
    val_dir = os.path.join(dataset_dir, "images", "val")
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    # 清空旧文件
    for d in [train_dir, val_dir]:
        for f in glob.glob(os.path.join(d, "*.*")):
            os.remove(f)

    for fp in train_imgs:
        shutil.copy2(fp, os.path.join(train_dir, os.path.basename(fp)))
    for fp in val_imgs:
        shutil.copy2(fp, os.path.join(val_dir, os.path.basename(fp)))

    ensure_dir(os.path.join(dataset_dir, "labels", "train"))
    ensure_dir(os.path.join(dataset_dir, "labels", "val"))

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    write_yaml(yaml_path, dataset_dir)

    print("\n[完成] 已生成数据集骨架（images 已就绪，labels 为空）。")
    print("下一步：建议运行 auto_label 自动标注（生成伪标签），或用 LabelImg 人工标注。")


# ---------------------------
# 模式：auto_label
# ---------------------------

def auto_label(dataset_dir: str,
               pose_model_path: str,
               imgsz: int,
               conf: float,
               wh_th: float,
               ang_th: float,
               headhip_rel_th: float,
               score_th: int,
               overwrite: bool,
               save_debug: bool):
    """
    自动标注 dataset/images/train & val -> dataset/labels/train & val
    只标注“躺姿判定通过”的帧，减少误标
    """
    train_img = os.path.join(dataset_dir, "images", "train")
    val_img = os.path.join(dataset_dir, "images", "val")
    train_lab = os.path.join(dataset_dir, "labels", "train")
    val_lab = os.path.join(dataset_dir, "labels", "val")

    if not os.path.isdir(train_img) or not os.path.isdir(val_img):
        raise RuntimeError("找不到 images/train 或 images/val，请先运行 make_dataset。")

    print(f"[信息] 自动标注使用 pose 模型：{pose_model_path}")
    print("[提示] 自动标注是伪标签：为了少误报，这里采用偏严格躺姿筛选（宁缺毋滥）。")

    lt, _ = auto_label_folder(train_img, train_lab, pose_model_path, imgsz, conf,
                              wh_th, ang_th, headhip_rel_th, score_th,
                              overwrite, save_debug)
    lv, _ = auto_label_folder(val_img, val_lab, pose_model_path, imgsz, conf,
                              wh_th, ang_th, headhip_rel_th, score_th,
                              overwrite, save_debug)

    print(f"\n[总结] 自动标注完成：train标注={lt}，val标注={lv}")
    if lt == 0 or lv == 0:
        print("【警告】标注数量为 0：说明筛选太严格或 pose 没检测到人。")
        print("你可以在 auto_label 里把阈值调松一些（例如 ang_th 降到 52，wh_th 降到 1.25，score_th 改 1）。")


# ---------------------------
# 模式：train（YOLO 检测训练）
# ---------------------------

def train_model(dataset_dir: str,
                model_pt: str,
                imgsz: int,
                epochs: int,
                batch: int,
                workers: int,
                lr0: float,
                patience: int,
                project: str,
                name: str):
    """单类摔倒检测训练，输出到 ./模型"""
    check_cuda_or_exit()

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise RuntimeError(f"找不到 data.yaml：{yaml_path}，请先运行 make_dataset。")

    lt, lv = check_labels_exist(dataset_dir)
    if lt == 0 or lv == 0:
        print("[错误] labels 缺失：YOLO 检测训练必须有框标注。")
        print(f"当前 labels 数量：train={lt}, val={lv}")
        print("请先运行 auto_label（自动标注）或手动标注，再训练。")
        sys.exit(1)

    ensure_dir(project)

    print(f"[信息] labels 数量：train={lt}, val={lv}")
    print(f"[信息] 开始训练：model={model_pt}, imgsz={imgsz}, epochs={epochs}, batch={batch}")
    print(f"[信息] 输出目录：{project}/{name}")

    model = YOLO(model_pt)

    model.train(
        data=yaml_path,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=0,            # 强制 GPU0
        workers=workers,
        lr0=lr0,
        cos_lr=True,
        patience=patience,
        project=project,
        name=name,
    )

    print("\n[完成] 训练结束。最优权重通常在：")
    print(os.path.join(project, name, "weights", "best.pt"))
    print("提示：best.pt 是“检测模型”。如果你的 view.py 用 Pose+规则/ML，还需要 pose 模型来算关键点。")


# ==========================================================
# ✅新增：事件分类器训练（只用 摔倒/ 单目录）
# 目标：区分“快摔倒(1)” vs “慢摔倒/慢慢倒下(0)”
# ==========================================================

def _pctl(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))

def build_clip_feature_from_window(window: List[Dict[str, float]], sample_fps: float) -> np.ndarray:
    """
    将“滑动窗口帧序列”转换为 19 维特征（与 view.py 的 build_clip_feature_from_window 对齐）
    window: 每帧一个 metrics dict（含 body_ang, wh_ratio, head_hip_v_rel, cy, bh, lying_like）
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
        _pctl(body_ang_seq, 95),

        float(np.mean(wh_ratio_seq)),
        float(np.max(wh_ratio_seq)),
        _pctl(wh_ratio_seq, 95),

        float(np.mean(headhip_seq)),
        float(np.min(headhip_seq)),

        float(np.mean(ang_speed)),
        float(np.max(ang_speed)),
        _pctl(ang_speed, 95),

        float(np.mean(vy_norm)),
        float(np.max(vy_norm)),
        _pctl(vy_norm, 95),

        float(np.mean(wh_speed)),
        float(np.max(wh_speed)),
        _pctl(wh_speed, 95),

        float(np.mean(lying_seq)),
        float(time_to_lying),
    ]
    return np.array(feats, dtype=np.float32)

def _list_videos(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    vids = []
    for e in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.m4v", "*.webm"):
        vids.extend(glob.glob(os.path.join(folder, e)))
    return sorted(vids)

def _pose_metrics_from_frame(model: YOLO,
                            frame: np.ndarray,
                            imgsz: int,
                            conf: float,
                            wh_th: float,
                            ang_th: float,
                            headhip_rel_th: float,
                            score_th: int) -> Optional[Dict[str, float]]:
    """
    单帧：用 pose 模型得到“最大人框”的 metrics
    返回 None 表示没检测到人或关键点不可用
    """
    res = model.predict(frame, imgsz=imgsz, conf=conf, device=0, verbose=False)[0]
    boxes = getattr(res, "boxes", None)
    kpts = getattr(res, "keypoints", None)
    if boxes is None or kpts is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    kxy = kpts.xy.cpu().numpy()
    if len(xyxy) == 0:
        return None

    # 取最大框的人（主角）
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    j = int(np.argmax(areas))

    m = calc_pose_metrics(kxy[j], xyxy[j])
    m["lying_like"] = is_lying_like(
        m,
        wh_th=wh_th,
        ang_th=ang_th,
        headhip_rel_th=headhip_rel_th,
        score_th=score_th
    )
    return m

def _extract_windows_from_video(video_path: str,
                                pose_model: YOLO,
                                imgsz: int,
                                conf: float,
                                sample_fps: float,
                                window_sec: float,
                                stride_sec: float,
                                max_windows_per_video: int,
                                wh_th: float,
                                ang_th: float,
                                headhip_rel_th: float,
                                score_th: int,
                                min_lying_ratio: float,
                                verbose_every: int = 300) -> Tuple[List[np.ndarray], List[float], List[float], List[float], List[float]]:
    """
    从单个视频中抽取窗口：
    返回：
      - X_feats: 19维特征
      - lying_mean: 躺姿比例（0~1）
      - max_ang_speed: 窗口内角速度最大值
      - max_vy_norm: 窗口内归一化下落速度最大值
      - max_wh_speed: 窗口内宽高比变化速度最大值
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[警告] 无法打开视频：{video_path}")
        return [], [], [], [], []

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1e-3:
        src_fps = 25.0

    # 每隔 step 帧取一帧用于 pose（近似 sample_fps）
    step = max(1, int(round(src_fps / max(0.1, sample_fps))))

    win_len = max(8, int(round(window_sec * sample_fps)))
    stride_len = max(1, int(round(stride_sec * sample_fps)))

    window: List[Dict[str, float]] = []
    X_feats: List[np.ndarray] = []
    lying_mean_list: List[float] = []
    max_ang_speed_list: List[float] = []
    max_vy_norm_list: List[float] = []
    max_wh_speed_list: List[float] = []

    frame_idx = 0
    sampled_idx = 0
    dropped = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        sampled_idx += 1

        m = _pose_metrics_from_frame(
            model=pose_model,
            frame=frame,
            imgsz=imgsz,
            conf=conf,
            wh_th=wh_th,
            ang_th=ang_th,
            headhip_rel_th=headhip_rel_th,
            score_th=score_th
        )

        if m is None:
            dropped += 1
            frame_idx += 1
            continue

        window.append(m)

        # 窗口满了就抽一次
        if len(window) >= win_len:
            clip = window[-win_len:]
            lying_seq = np.array([1.0 if w.get("lying_like", False) else 0.0 for w in clip], dtype=np.float32)
            lying_mean = float(np.mean(lying_seq))

            # 只保留“确实进入躺姿”的窗口（否则慢摔/快摔都不成立）
            if lying_mean >= float(min_lying_ratio):
                # 计算速度突变指标
                dt = 1.0 / max(1e-6, float(sample_fps))
                body_ang_seq = np.array([w["body_ang"] for w in clip], dtype=np.float32)
                wh_ratio_seq = np.array([w["wh_ratio"] for w in clip], dtype=np.float32)
                cy_seq = np.array([w["cy"] for w in clip], dtype=np.float32)
                bh_seq = np.array([w["bh"] for w in clip], dtype=np.float32)

                ang_speed = np.diff(body_ang_seq) / dt
                wh_speed = np.diff(wh_ratio_seq) / dt
                vy = np.diff(cy_seq) / dt
                vy_norm = vy / (bh_seq[1:] + 1e-6)

                max_ang_speed = float(np.max(ang_speed)) if len(ang_speed) else 0.0
                max_wh_speed = float(np.max(wh_speed)) if len(wh_speed) else 0.0
                max_vy_norm = float(np.max(vy_norm)) if len(vy_norm) else 0.0

                feat = build_clip_feature_from_window(clip, sample_fps=sample_fps)
                X_feats.append(feat)
                lying_mean_list.append(lying_mean)
                max_ang_speed_list.append(max_ang_speed)
                max_vy_norm_list.append(max_vy_norm)
                max_wh_speed_list.append(max_wh_speed)

            # 步进滑窗
            window = window[stride_len:]

            if len(X_feats) >= max_windows_per_video:
                break

        if sampled_idx % verbose_every == 0:
            print(f"[抽样] {os.path.basename(video_path)} 已采样帧={sampled_idx} 掉帧(无人)={dropped} 窗口数={len(X_feats)}")

        frame_idx += 1

    cap.release()
    return X_feats, lying_mean_list, max_ang_speed_list, max_vy_norm_list, max_wh_speed_list

def train_event_clf_single_dir(fall_dir: str,
                               pose_model_path: str,
                               imgsz: int,
                               conf: float,
                               sample_fps: float,
                               window_sec: float,
                               stride_sec: float,
                               max_windows_per_video: int,
                               # 躺姿判定阈值（偏严格，减少误报）
                               wh_th: float,
                               ang_th: float,
                               headhip_rel_th: float,
                               score_th: int,
                               # 只保留躺姿比例超过该阈值的窗口
                               min_lying_ratio: float,
                               # 初始“快摔倒”阈值（如果分不出来会自动用分位数自适应）
                               ang_speed_th: float,
                               vy_norm_th: float,
                               wh_speed_th: float,
                               # 自适应切分快摔倒的分位数（例如 80% 表示 top20% 当快摔倒）
                               fast_quantile: float,
                               out_joblib: str,
                               test_ratio: float,
                               seed: int):
    """
    事件分类器训练（单目录版本）：
    - 只用 fall_dir 下的视频
    - 自动从“窗口”上打伪标签：快摔倒(1) / 慢摔倒(0)
    - 输出 sklearn 模型到 joblib（供 view.py 加载）
    """
    check_cuda_or_exit()

    # sklearn/joblib 依赖检查
    try:
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        print("【错误】缺少事件分类训练依赖：scikit-learn / joblib")
        print("请执行：pip install scikit-learn joblib")
        print("原始错误：", e)
        sys.exit(1)

    videos = _list_videos(fall_dir)
    if len(videos) == 0:
        raise RuntimeError(f"找不到视频：{fall_dir}\n请把你的摔倒视频放到该目录（mp4/avi/mkv/mov...）。")

    print(f"[信息] 使用单目录摔倒视频：{fall_dir}，视频数={len(videos)}")
    print(f"[信息] 使用 pose 模型：{pose_model_path}")
    print(f"[信息] 抽样参数：sample_fps={sample_fps}, window_sec={window_sec}, stride_sec={stride_sec}, 每视频最多窗口={max_windows_per_video}")
    print(f"[信息] 伪标签：躺姿比例>= {min_lying_ratio} 才参与；快摔倒阈值（初始）angSp>{ang_speed_th}, vyN>{vy_norm_th}, whSp>{wh_speed_th}")
    print("[提示] 如果初始阈值分不出快/慢，本脚本会自动按分位数切分，确保能训练出二分类模型。")

    pose_model = YOLO(pose_model_path)

    X_all: List[np.ndarray] = []
    lying_mean_all: List[float] = []
    angSp_all: List[float] = []
    vyN_all: List[float] = []
    whSp_all: List[float] = []

    for vp in videos:
        X, ly, aS, vN, wS = _extract_windows_from_video(
            video_path=vp,
            pose_model=pose_model,
            imgsz=imgsz,
            conf=conf,
            sample_fps=sample_fps,
            window_sec=window_sec,
            stride_sec=stride_sec,
            max_windows_per_video=max_windows_per_video,
            wh_th=wh_th,
            ang_th=ang_th,
            headhip_rel_th=headhip_rel_th,
            score_th=score_th,
            min_lying_ratio=min_lying_ratio
        )
        X_all.extend(X)
        lying_mean_all.extend(ly)
        angSp_all.extend(aS)
        vyN_all.extend(vN)
        whSp_all.extend(wS)
        print(f"[窗口] {os.path.basename(vp)} -> 有效窗口 {len(X)}")

    if len(X_all) < 60:
        print("【警告】有效窗口样本偏少，joblib 可能不稳定。建议增加视频数量或调小 min_lying_ratio / 增大 max_windows_per_video。")

    X_arr = np.stack(X_all, axis=0).astype(np.float32)

    # 计算“快摔倒打分”：三种速度突变取最大（越大越像快摔倒）
    # score=1 表示达到初始阈值；>1 表示超过阈值；<1 表示更慢
    angSp = np.array(angSp_all, dtype=np.float32)
    vyN = np.array(vyN_all, dtype=np.float32)
    whSp = np.array(whSp_all, dtype=np.float32)

    score = np.maximum.reduce([
        angSp / max(1e-6, float(ang_speed_th)),
        vyN / max(1e-6, float(vy_norm_th)),
        whSp / max(1e-6, float(wh_speed_th)),
    ])

    # 先用“是否超过阈值”作为快摔倒伪标签
    y0 = (score >= 1.0).astype(np.int64)

    # 如果只有一个类别（全是快 或 全是慢），就自适应切分
    uniq = np.unique(y0)
    if len(uniq) < 2:
        # 用分位数把 top(1-fast_quantile) 当快摔倒
        fq = float(fast_quantile)
        fq = max(0.55, min(0.95, fq))  # 防止切得太极端
        thr = float(np.quantile(score, fq))
        y0 = (score >= thr).astype(np.int64)
        uniq = np.unique(y0)
        print(f"【提示】初始阈值无法分出快/慢，已启用分位数切分：quantile={fq}，score阈值={thr:.3f}")

    # 仍然只有一个类，最后兜底：用中位数切分
    if len(np.unique(y0)) < 2:
        thr = float(np.median(score))
        y0 = (score >= thr).astype(np.int64)
        print(f"【提示】分位数仍无法分出两类，已用中位数切分：score阈值={thr:.3f}")

    # 检查每类数量
    n1 = int(np.sum(y0 == 1))
    n0 = int(np.sum(y0 == 0))
    print(f"[信息] 伪标签统计：快摔倒(1)={n1}，慢摔倒(0)={n0}")

    if min(n0, n1) < 10:
        print("【警告】某一类样本过少，训练可能不稳。你可以：")
        print("  1) 降低 min_lying_ratio（让更多窗口进入训练）")
        print("  2) 调整 ang_speed_th / vy_norm_th / wh_speed_th 或 fast_quantile")
        print("  3) 增加视频数量/时长")

    # 划分训练/测试
    Xtr, Xte, ytr, yte = train_test_split(
        X_arr, y0,
        test_size=float(test_ratio),
        random_state=int(seed),
        shuffle=True,
        stratify=y0 if (min(n0, n1) >= 2) else None
    )

    # 训练：LogisticRegression（有 predict_proba，适配 view.py）
    clf = LogisticRegression(max_iter=2500)
    clf.fit(Xtr, ytr)

    # 评估输出
    pred = clf.predict(Xte)
    print("\n========== 事件分类器评估（测试集，伪标签）==========")
    print(classification_report(yte, pred, digits=4))
    try:
        cm = confusion_matrix(yte, pred)
        print("混淆矩阵：")
        print(cm)
    except Exception:
        pass

    # 保存
    out_dir = os.path.dirname(out_joblib)
    if out_dir:
        ensure_dir(out_dir)
    import joblib
    joblib.dump(clf, out_joblib)
    print("\n[完成] 已保存事件分类器（joblib）：")
    print(out_joblib)
    print("下一步：打开 view.py，算法模式选择“训练模型/混合”，并选择该 joblib 文件。")


# ---------------------------
# 命令行参数（可选）
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="train.py（对话式设置 + make_dataset/auto_label/train/train_event_clf）")
    ap.add_argument("--mode", type=str,
                    choices=["make_dataset", "auto_label", "train", "train_event_clf"],
                    default="",
                    help="不传则进入对话式设置")

    ap.add_argument("--dataset_dir", type=str, default="")
    ap.add_argument("--fall_raw", type=str, default="")
    ap.add_argument("--extract_fps", type=float, default=-1)
    ap.add_argument("--val_ratio", type=float, default=-1)

    # auto_label
    ap.add_argument("--pose_model", type=str, default="")
    ap.add_argument("--al_imgsz", type=int, default=-1)
    ap.add_argument("--al_conf", type=float, default=-1)
    ap.add_argument("--wh_th", type=float, default=-1)
    ap.add_argument("--ang_th", type=float, default=-1)
    ap.add_argument("--hh_th", type=float, default=-1)
    ap.add_argument("--score_th", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_debug", action="store_true")

    # train（YOLO检测）
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--imgsz", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--lr0", type=float, default=-1)
    ap.add_argument("--patience", type=int, default=-1)
    ap.add_argument("--project", type=str, default="")
    ap.add_argument("--name", type=str, default="")

    # ✅ train_event_clf（单目录）
    ap.add_argument("--clf_out", type=str, default="")
    ap.add_argument("--clf_imgsz", type=int, default=-1)
    ap.add_argument("--clf_conf", type=float, default=-1)
    ap.add_argument("--sample_fps", type=float, default=-1)
    ap.add_argument("--window_sec", type=float, default=-1)
    ap.add_argument("--stride_sec", type=float, default=-1)
    ap.add_argument("--max_windows_per_video", type=int, default=-1)
    ap.add_argument("--min_lying_ratio", type=float, default=-1)

    ap.add_argument("--ang_speed_th", type=float, default=-1)
    ap.add_argument("--vy_norm_th", type=float, default=-1)
    ap.add_argument("--wh_speed_th", type=float, default=-1)
    ap.add_argument("--fast_quantile", type=float, default=-1)

    ap.add_argument("--test_ratio", type=float, default=-1)
    ap.add_argument("--seed", type=int, default=-1)

    return ap.parse_args()


# ---------------------------
# 对话式流程（回车=默认）
# ---------------------------

def interactive_flow():
    print("\n====== train.py 对话式设置（回车=默认值）======")
    print(f"[默认] 摔倒素材目录：{DEFAULT_FALL_RAW}")
    print(f"[默认] 数据集目录：{DEFAULT_DATASET_DIR}")
    print(f"[默认] 训练输出目录：{DEFAULT_PROJECT_DIR}")
    print(f"[默认] 自动标注 pose 模型：{DEFAULT_POSE_MODEL}")
    print(f"[默认] 事件分类 joblib 输出：{DEFAULT_JOBLIB_OUT}\n")

    mode = ask_choice("请选择模式",
                      ["make_dataset", "auto_label", "train", "train_event_clf"],
                      "make_dataset")

    if mode == "make_dataset":
        dataset_dir = ask_str("数据集输出目录 dataset_dir", DEFAULT_DATASET_DIR)
        fall_raw = ask_str("摔倒素材目录 fall_raw（视频/图片）", DEFAULT_FALL_RAW)
        extract_fps = ask_float("视频抽帧 FPS（每秒抽几帧）", 5.0, minv=0.5, maxv=30.0)
        val_ratio = ask_float("验证集比例 val_ratio", 0.2, minv=0.05, maxv=0.5)
        make_dataset(dataset_dir, fall_raw, extract_fps, val_ratio)
        return

    if mode == "auto_label":
        dataset_dir = ask_str("数据集目录 dataset_dir（包含 images/train,val）", DEFAULT_DATASET_DIR)
        pose_model = ask_str("pose 模型路径（用于自动标注）", DEFAULT_POSE_MODEL)

        al_imgsz = ask_int("自动标注 imgsz", 960, minv=320, maxv=1536)
        al_conf = ask_float("自动标注 conf", 0.35, minv=0.05, maxv=0.95)

        wh_th = ask_float("躺姿筛选：宽高比阈值 wh_th", 1.35, minv=1.05, maxv=3.0)
        ang_th = ask_float("躺姿筛选：角度阈值 ang_th（>表示更水平）", 58.0, minv=35.0, maxv=89.0)
        hh_th = ask_float("躺姿筛选：头-髋相对高度 hh_th（<表示更躺）", 0.16, minv=0.05, maxv=0.5)
        score_th = ask_int("躺姿筛选：满足条件个数 score_th（1~3）", 2, minv=1, maxv=3)

        overwrite = ask_choice("是否覆盖已有 labels？", ["y", "n"], "n") == "y"
        save_debug = ask_choice("是否保存调试图（会生成 _debug_autolabel）？", ["y", "n"], "y") == "y"

        auto_label(dataset_dir, pose_model, al_imgsz, al_conf,
                   wh_th, ang_th, hh_th, score_th,
                   overwrite=overwrite, save_debug=save_debug)
        return

    if mode == "train":
        dataset_dir = ask_str("数据集目录 dataset_dir（包含 data.yaml 与 labels）", DEFAULT_DATASET_DIR)
        model_pt = ask_str("起始权重 model（建议 yolo11s.pt 或 yolo11m.pt）", "yolo11s.pt")
        imgsz = ask_int("训练 imgsz（640~1280）", 960, minv=320, maxv=1536)
        epochs = ask_int("训练 epochs（轮数）", 140, minv=20, maxv=500)
        batch = ask_int("batch（显存不够就减）", 16, minv=1, maxv=512)
        workers = ask_int("workers（数据加载线程）", 4, minv=0, maxv=32)
        lr0 = ask_float("初始学习率 lr0", 0.01, minv=1e-5, maxv=0.1)
        patience = ask_int("早停 patience（不提升就停止）", 30, minv=5, maxv=200)
        project = ask_str("输出目录 project（训练结果放这里）", DEFAULT_PROJECT_DIR)
        name = ask_str("实验名 name", "exp")
        train_model(dataset_dir, model_pt, imgsz, epochs, batch, workers, lr0, patience, project, name)
        return

    # ✅ train_event_clf（只用 摔倒/）
    fall_dir = ask_str("摔倒视频目录 fall_dir（里面包含慢摔倒/快摔倒）", DEFAULT_FALL_RAW)
    pose_model = ask_str("pose 模型路径（用于抽特征）", DEFAULT_POSE_MODEL)

    clf_imgsz = ask_int("抽特征 pose imgsz", 960, minv=320, maxv=1536)
    clf_conf = ask_float("抽特征 pose conf", 0.35, minv=0.05, maxv=0.95)

    sample_fps = ask_float("抽样帧率 sample_fps（建议10）", 10.0, minv=2.0, maxv=30.0)
    window_sec = ask_float("窗口长度 window_sec（秒，建议2.0）", 2.0, minv=0.8, maxv=6.0)
    stride_sec = ask_float("窗口步进 stride_sec（秒，建议0.5）", 0.5, minv=0.1, maxv=3.0)
    max_windows_per_video = ask_int("每个视频最多窗口 max_windows_per_video", 160, minv=20, maxv=4000)

    # 躺姿阈值（偏严格）
    wh_th = ask_float("躺姿筛选：宽高比阈值 wh_th", 1.35, minv=1.05, maxv=3.0)
    ang_th = ask_float("躺姿筛选：角度阈值 ang_th（>表示更水平）", 58.0, minv=35.0, maxv=89.0)
    hh_th = ask_float("躺姿筛选：头-髋相对高度 hh_th（<表示更躺）", 0.16, minv=0.05, maxv=0.5)
    score_th = ask_int("躺姿筛选：满足条件个数 score_th（1~3）", 2, minv=1, maxv=3)

    min_lying_ratio = ask_float("窗口躺姿比例下限 min_lying_ratio（建议0.35~0.55）", 0.45, minv=0.1, maxv=0.95)

    # 初始快摔倒阈值（如果分不出来会自动按分位数切分）
    ang_speed_th = ask_float("快摔倒阈值：角速度 ang_speed_th（°/s）", 190.0, minv=20.0, maxv=800.0)
    vy_norm_th = ask_float("快摔倒阈值：归一化下落速度 vy_norm_th（框高/s）", 1.45, minv=0.2, maxv=6.0)
    wh_speed_th = ask_float("快摔倒阈值：宽高比速度 wh_speed_th（/s）", 1.80, minv=0.2, maxv=8.0)

    fast_quantile = ask_float("自适应切分分位数 fast_quantile（0.6~0.9，建议0.8）", 0.80, minv=0.6, maxv=0.9)

    test_ratio = ask_float("测试集比例 test_ratio", 0.25, minv=0.1, maxv=0.5)
    seed = ask_int("随机种子 seed", 42, minv=0, maxv=999999)

    out_joblib = ask_str("joblib 输出路径 out_joblib", DEFAULT_JOBLIB_OUT)

    train_event_clf_single_dir(
        fall_dir=fall_dir,
        pose_model_path=pose_model,
        imgsz=clf_imgsz,
        conf=clf_conf,
        sample_fps=sample_fps,
        window_sec=window_sec,
        stride_sec=stride_sec,
        max_windows_per_video=max_windows_per_video,
        wh_th=wh_th,
        ang_th=ang_th,
        headhip_rel_th=hh_th,
        score_th=score_th,
        min_lying_ratio=min_lying_ratio,
        ang_speed_th=ang_speed_th,
        vy_norm_th=vy_norm_th,
        wh_speed_th=wh_speed_th,
        fast_quantile=fast_quantile,
        out_joblib=out_joblib,
        test_ratio=test_ratio,
        seed=seed
    )


def main():
    args = parse_args()

    # 不传 mode：对话式
    if not args.mode:
        interactive_flow()
        return

    if args.mode == "make_dataset":
        dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
        fall_raw = args.fall_raw or DEFAULT_FALL_RAW
        extract_fps = args.extract_fps if args.extract_fps > 0 else 5.0
        val_ratio = args.val_ratio if args.val_ratio > 0 else 0.2
        make_dataset(dataset_dir, fall_raw, extract_fps, val_ratio)
        return

    if args.mode == "auto_label":
        dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
        pose_model = args.pose_model or DEFAULT_POSE_MODEL
        al_imgsz = args.al_imgsz if args.al_imgsz > 0 else 960
        al_conf = args.al_conf if args.al_conf > 0 else 0.35
        wh_th = args.wh_th if args.wh_th > 0 else 1.35
        ang_th = args.ang_th if args.ang_th > 0 else 58.0
        hh_th = args.hh_th if args.hh_th > 0 else 0.16
        score_th = args.score_th if args.score_th > 0 else 2
        auto_label(dataset_dir, pose_model, al_imgsz, al_conf,
                   wh_th, ang_th, hh_th, score_th,
                   overwrite=bool(args.overwrite),
                   save_debug=bool(args.save_debug))
        return

    if args.mode == "train":
        dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
        model_pt = args.model or "yolo11s.pt"
        imgsz = args.imgsz if args.imgsz > 0 else 960
        epochs = args.epochs if args.epochs > 0 else 140
        batch = args.batch if args.batch > 0 else 16
        workers = args.workers if args.workers >= 0 else 4
        lr0 = args.lr0 if args.lr0 > 0 else 0.01
        patience = args.patience if args.patience > 0 else 30
        project = args.project or DEFAULT_PROJECT_DIR
        name = args.name or "exp"
        train_model(dataset_dir, model_pt, imgsz, epochs, batch, workers, lr0, patience, project, name)
        return

    # ✅ train_event_clf（单目录）
    fall_dir = args.fall_raw or DEFAULT_FALL_RAW
    pose_model = args.pose_model or DEFAULT_POSE_MODEL

    clf_out = args.clf_out or DEFAULT_JOBLIB_OUT
    clf_imgsz = args.clf_imgsz if args.clf_imgsz > 0 else 960
    clf_conf = args.clf_conf if args.clf_conf > 0 else 0.35

    sample_fps = args.sample_fps if args.sample_fps > 0 else 10.0
    window_sec = args.window_sec if args.window_sec > 0 else 2.0
    stride_sec = args.stride_sec if args.stride_sec > 0 else 0.5
    max_windows_per_video = args.max_windows_per_video if args.max_windows_per_video > 0 else 160

    wh_th = args.wh_th if args.wh_th > 0 else 1.35
    ang_th = args.ang_th if args.ang_th > 0 else 58.0
    hh_th = args.hh_th if args.hh_th > 0 else 0.16
    score_th = args.score_th if args.score_th > 0 else 2

    min_lying_ratio = args.min_lying_ratio if args.min_lying_ratio > 0 else 0.45

    ang_speed_th = args.ang_speed_th if args.ang_speed_th > 0 else 190.0
    vy_norm_th = args.vy_norm_th if args.vy_norm_th > 0 else 1.45
    wh_speed_th = args.wh_speed_th if args.wh_speed_th > 0 else 1.80
    fast_quantile = args.fast_quantile if args.fast_quantile > 0 else 0.80

    test_ratio = args.test_ratio if args.test_ratio > 0 else 0.25
    seed = args.seed if args.seed >= 0 else 42

    train_event_clf_single_dir(
        fall_dir=fall_dir,
        pose_model_path=pose_model,
        imgsz=clf_imgsz,
        conf=clf_conf,
        sample_fps=sample_fps,
        window_sec=window_sec,
        stride_sec=stride_sec,
        max_windows_per_video=max_windows_per_video,
        wh_th=wh_th,
        ang_th=ang_th,
        headhip_rel_th=hh_th,
        score_th=score_th,
        min_lying_ratio=min_lying_ratio,
        ang_speed_th=ang_speed_th,
        vy_norm_th=vy_norm_th,
        wh_speed_th=wh_speed_th,
        fast_quantile=fast_quantile,
        out_joblib=clf_out,
        test_ratio=test_ratio,
        seed=seed
    )


if __name__ == "__main__":
    main()
