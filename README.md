# 基于 YOLOv11 Pose 的多路视频人体摔倒检测与实时告警系统

## 项目简介

本项目是一个面向监控场景的人体摔倒检测系统，采用 **YOLOv11 Pose + OpenCV** 构建实时识别流程，并在此基础上融合 **姿态规则判别** 与 **事件分类器**，实现对“快速摔倒”和“缓慢躺下”等相似行为的区分。系统支持多路摄像头输入、目标跟踪、实时画面展示以及 pushplus 微信消息推送。

项目训练侧提供完整的数据处理与训练流程，包括：
- 从视频中抽帧构建数据集
- 自动生成摔倒伪标签
- 训练单类 `fall` 检测模型
- 训练“摔倒 vs 躺下”事件分类器 `fall_event_clf.joblib`

---

## 项目特点

- 基于 **YOLOv11 Pose** 提取人体关键点与姿态信息，用于摔倒行为识别
- 结合 **姿态角度、宽高比、头髋相对高度** 等特征进行单帧躺姿判定
- 引入 **规则法快速倒下事件检测**，通过角速度、归一化下落速度、宽高比变化速度增强摔倒识别能力
- 引入 **训练好的事件分类器**，区分“摔倒”与“慢慢躺下”，降低误报
- 支持 **规则 / 训练模型 / 混合模式** 三种推理策略
- 支持 **多路摄像头输入**、自动扫描本地摄像头、视频拼接显示与告警推送

---

## 项目结构

```text
.
├── train.py                  # 数据集制作、自动标注、检测模型训练、事件分类器训练
├── view.py                   # 多路摄像头实时摔倒检测与告警
├── dataset/                  # 训练数据集目录
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── all_fall/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── 模型/
│   └── exp/
│       └── weights/
│           └── best.pt       # 检测模型输出
├── 摔倒/                     # 原始摔倒视频或图片素材
├── yolo11n-pose.pt           # 自动标注/姿态推理用 pose 模型
└── fall_event_clf.joblib     # 事件分类器输出
```

---

## 环境要求

- Python 3.10 及以上
- NVIDIA GPU + CUDA
- Windows / Linux

> 本项目训练与实时推理默认要求 CUDA 环境。

---

## 安装依赖

推荐安装命令：

```bash
pip install ultralytics opencv-python pyyaml numpy requests scikit-learn joblib
pip install lapx
```

如需启用 GPU，请确保已安装支持 CUDA 的 PyTorch，并且 `nvidia-smi` 能正常识别显卡。

---

## 训练流程

### 1. 生成数据集

将原始摔倒视频或图片放入 `摔倒/` 目录，然后运行：

```bash
python train.py
```

在交互界面中选择：

```text
make_dataset
```

该模式会抽帧或复制图片，并自动生成：

```text
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
dataset/data.yaml
```

---

### 2. 自动标注

继续运行：

```bash
python train.py
```

选择：

```text
auto_label
```

该模式使用 pose 模型对训练集和验证集自动生成伪标签，只对满足“躺姿判定”的图像写入 `fall` 标签。

---

### 3. 训练检测模型

继续运行：

```bash
python train.py
```

选择：

```text
train
```

训练完成后，最优权重通常输出在：

```text
./模型/exp/weights/best.pt
```

---

### 4. 训练事件分类器

继续运行：

```bash
python train.py
```

选择：

```text
train_event_clf
```

训练完成后会生成：

```text
./fall_event_clf.joblib
```

---

## 实时检测

运行实时检测程序：

```bash
python view.py
```

程序支持：
- 多路摄像头输入
- YOLO Pose 实时推理
- 规则法摔倒判定
- 训练模型二次判别
- 混合模式识别
- pushplus 微信消息推送

---

## 推理模式说明

### 规则模式
仅依赖姿态规则和快速倒下事件判断。

### 训练模型模式
仅依赖训练好的事件分类器输出摔倒概率。

### 混合模式
结合规则法与分类器联合判定，整体更稳健，适合实际演示与部署。

---

## 告警功能

系统集成 pushplus 消息推送，可在检测到疑似摔倒后发送微信提醒，告警内容包括：
- 摄像头编号
- 检测时间
- 目标 ID
- 躺姿连续帧数
- 姿态角度
- 角速度
- 下落速度
- 分类器摔倒概率

若未填写有效 token，则程序会在控制台打印模拟告警信息。

---

## 推荐使用流程

```text
1. 准备原始摔倒视频/图片素材
2. 运行 train.py -> make_dataset
3. 运行 train.py -> auto_label
4. 运行 train.py -> train
5. 运行 train.py -> train_event_clf
6. 运行 view.py 进行实时检测与告警
```

---

## 应用场景

- 宿舍/居家老人看护
- 医院病房安全监测
- 智慧养老
- 公共场所异常行为监测
- 校园与社区安防

---

## 后续优化方向

- 增加人工标注数据，减少伪标签误差
- 引入更多正常行为负样本，提高泛化能力
- 优化事件分类器训练策略，增强“摔倒 / 躺下”区分效果
- 增加录像回放、事件日志与可视化管理功能
- 支持 Web 端部署与远程监控

---

## 致谢

本项目基于 Ultralytics YOLO、OpenCV、Tkinter、scikit-learn 等开源工具完成训练、推理与界面构建。
