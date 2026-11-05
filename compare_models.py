"""直接运行，评估 checkpoints 中的模型并在 results 中保存图表。"""
import csv
import os
from pathlib import Path

# 可自行修改要评估的模型。
MODELS = ["LSTM", "GRU", "TCN"]
BATCH_SIZE = 128
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_ipn import Classifier, ROOT, dataset

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]
NUM_CLASSES = 15


def calculate_metrics(confusion):
    """行是真实类别，列是预测类别；分母为零时，该类指标计为零。"""
    true_positive = np.diag(confusion).astype(float)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    if confusion.sum() == 0:
        raise ValueError("测试集为空")
    precision = np.divide(true_positive, predicted, out=np.zeros(NUM_CLASSES), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(NUM_CLASSES), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(NUM_CLASSES), where=(precision + recall) > 0)
    return [true_positive.sum() / confusion.sum(), precision.mean(), recall.mean(), f1.mean()]


@torch.inference_mode()
def evaluate(kind, loader):
    model = Classifier(kind)
    weights = torch.load(ROOT / "checkpoints" / f"{kind}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for clips, labels in loader:
        predictions = model(clips).argmax(dim=1).numpy()
        np.add.at(confusion, (labels.numpy(), predictions), 1)
    return calculate_metrics(confusion)


def main():
    torch.set_num_threads(2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset("test"), batch_size=BATCH_SIZE, shuffle=False)
    scores = np.array([evaluate(kind, loader) for kind in MODELS])
    with (OUTPUT_DIR / "model_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "Test_samples"])
        for kind, values in zip(MODELS, scores):
            writer.writerow([kind, *values, len(loader.dataset)])
            print(kind, " | ".join(f"{name}={value:.2%}" for name, value in zip(METRICS, values)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(MODELS))]
    for metric_index, ax in enumerate(axes.flat):
        bars = ax.bar(MODELS, scores[:, metric_index] * 100, color=colors, width=0.55)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=11, padding=4)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
        ax.set_title(METRICS[metric_index], fontsize=14)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Gesture recognition model comparison · {len(loader.dataset):,} test clips\n"
        "Precision / Recall / F1-score: macro average over 15 classes",
        fontsize=14,
    )
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "model_comparison.svg")
    plt.close(fig)
    print(f"图表和指标已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
