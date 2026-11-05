import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/ipn_original"
OUTPUT = ROOT / "data/processed"
STEPS = 32


def prepare():
    annotations = SOURCE / "annotations"
    seen = set()
    records = []
    for split in ("train", "test"):
        clips, labels = [], []
        for video in (annotations / "splits" / f"{split}_list.txt").read_text().splitlines():
            video = video.strip()
            if not video:
                continue
            if video in seen:
                raise ValueError(f"Repeated video: {video}")
            seen.add(video)
            frames = json.loads((SOURCE / "landmarks/landmarks_mp_hands" / f"{video}.json").read_bytes()) #读取标注
            sequence = np.zeros((len(frames), 63), np.float32) #创建矩阵[x0 y0 z0,
                                                                              #x1 y1 z1, ... x20 y20 z20] 63维
            for i in range(len(frames)):
                hands = frames[f"{video}_{i+1:06}.jpg"]["hand_landmarks"]
                if hands:
                    points = hands[0]["landmarks"]
                    sequence[i] = np.asarray([points[str(j)] for j in range(21)]).ravel()#将形状为 (21, 3) 的数据展平成 (63,)。
            for row in (annotations / "temporal_annotations" / f"{video}.txt").read_text().splitlines():# 动作标注截取手势
                label, start, end = map(int, row.split(","))
                if not (0 <= label < 15 and 1 <= start <= end <= len(sequence)):
                    # label 处于 0～14。start 至少为 1。end 不小于 start。end 不超过视频总帧数。
                    raise ValueError((video, row, len(sequence)))
                idx = np.linspace(start-1, end-1, STEPS).round().astype(int) #统一转化成32，从其中均匀选择 32 帧。
                records.append(dict(split=split, index=len(labels), video=video, start=start, end=end, label=label))
                clips.append(sequence[idx])
                labels.append(label)
        np.savez_compressed(OUTPUT / split / "clips.npz", x=np.stack(clips), y=labels) #保存成模型
        print(split, len(labels), flush=True)
    (OUTPUT / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    prepare()
