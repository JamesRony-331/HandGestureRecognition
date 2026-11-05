import torch
from torch.utils.data import DataLoader

from train_ipn import Classifier, ROOT, dataset

# 直接运行评估全部模型，也可改成 ["GRU"] 等单模型列表。
MODELS = ["LSTM", "GRU", "TCN"]


@torch.inference_mode() #在模型推理时关闭自动梯度计算，并进一步减少部分张量管理开销，从而提高推理效率。
def evaluate(kind, test_loader):
    model = Classifier(kind)
    weights = torch.load(ROOT / "checkpoints" / f"{kind}.pt", map_location="cpu")
    model.load_state_dict(weights)
    model.eval()

    correct = 0
    total = 0
    for clips, labels in test_loader:
        predictions = model(clips).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
    if not total:
        raise ValueError("测试集为空，无法计算正确率")
    print(f"{kind}：正确 {correct}/{total}，测试集正确率 {correct / total:.2%}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_loader = DataLoader(dataset("test"), batch_size=128, shuffle=False)
    for kind in MODELS:
        evaluate(kind, test_loader)
