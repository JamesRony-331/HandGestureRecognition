
import cv2
import mediapipe as mp
import time
from pathlib import Path

# 在这里选择手势分类模型，可填 "GRU"、"LSTM" 或 "TCN"
GESTURE_MODEL = "TCN"
PREVIEW_SIZE = (960, 720)  # 预览窗口宽、高（像素）
INFERENCE_INTERVAL = 5  # 手势分类间隔（帧），1 表示逐帧推理

# 模型路径
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# 手部骨架连接关系
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand(frame, landmarks):
    """绘制21个关键点及手部骨架"""

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        points.append((x, y))

    # 绘制关节连接线
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # 绘制21个关键点和编号
    for i, (x, y) in enumerate(points):
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )


GESTURE_NAMES = {
    "NOG": "非手势", "P1F": "单指指向", "P2F": "双指指向",
    "C1F": "单指点击", "C2F": "双指点击", "TUP": "向上抛动",
    "TDN": "向下抛动", "TLT": "向左抛动", "TRT": "向右抛动",
    "OP2": "两次张开", "2C1": "单指双击", "2C2": "双指双击",
    "ZIN": "放大", "ZOT": "缩小", "GRB": "抓取",
}


def main():
    import argparse
    import csv
    from collections import deque

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["LSTM", "GRU", "TCN"], default=GESTURE_MODEL)
    kind = parser.parse_args().model
    if kind:
        import numpy as np
        import torch
        from train_ipn import Classifier
        from prepare_ipn import STEPS
        model = Classifier(kind)
        model.load_state_dict(torch.load(BASE_DIR / "checkpoints" / f"{kind}.pt", map_location="cpu"))
        model.eval()
        if type(INFERENCE_INTERVAL) is not int or INFERENCE_INTERVAL < 1:
            raise ValueError("INFERENCE_INTERVAL 必须是正整数")
        frames_until_prediction = 0
        gesture = 0
        history = deque(maxlen=STEPS)
        with (BASE_DIR / "data/ipn_original/annotations/classes.txt").open() as labels:
            names = {int(row["id"]): GESTURE_NAMES[row["code"]] for row in csv.DictReader(labels)}
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 28)
        badges = {}
        for number, name in names.items():
            badge = Image.new("RGB", (150, 40))
            ImageDraw.Draw(badge).text((4, 0), name, font=font, fill="yellow")
            badges[number] = cv2.cvtColor(np.asarray(badge), cv2.COLOR_RGB2BGR)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"未找到模型文件: {MODEL_PATH}"
        )

    # 初始化手部检测模型
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.monotonic()
    last_time = start_time
    last_timestamp = -1

    try:
        cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Detection", *PREVIEW_SIZE)
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("摄像头读取失败")
                    break

                # 镜像显示
                if not kind:
                    frame = cv2.flip(frame, 1)

                # BGR 转 RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # 创建 MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # 生成严格递增的毫秒时间戳
                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                # 检测手部关键点
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                # 绘制每只手的骨架
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks)

                if kind:
                    points = result.hand_landmarks
                    vector = [v for p in points[0] for v in (p.x, p.y, p.z)] if points else np.zeros(63)
                    history.append(vector)
                    if len(history) == STEPS:
                        if not points:
                            gesture = 0
                            frames_until_prediction = 0
                        elif frames_until_prediction == 0:
                            with torch.no_grad():
                                batch = torch.from_numpy(np.asarray(history, dtype=np.float32))[None]
                                gesture = model(batch).argmax(1).item()
                            frames_until_prediction = INFERENCE_INTERVAL - 1
                        else:
                            frames_until_prediction -= 1
                        region = frame[85:125, 20:170]
                        region[:] = badges[gesture][:region.shape[0], :region.shape[1]]

                # 显示检测到的手数
                hand_count = len(result.hand_landmarks)

                cv2.putText(
                    frame,
                    f"Hands: {hand_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # 计算实时FPS
                now = time.monotonic()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Hand Detection", frame)

                # 按Q退出
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
