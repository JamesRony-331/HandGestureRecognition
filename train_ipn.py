import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from prepare_ipn import ROOT, OUTPUT

epochs=400
class Classifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "TCN":
            layers = []
            for i, dilation in enumerate((1, 2, 4)):
                layers.extend([nn.Conv1d(63 if i == 0 else 64, 64, 3, padding=dilation, dilation=dilation), nn.ReLU()])
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = getattr(nn, kind)(63, 64, batch_first=True)
        self.head = nn.Linear(64, 15)

    def forward(self, x):
        if self.kind == "TCN":
            h = self.encoder(x.transpose(1, 2)).mean(2)
        else:
            h = self.encoder(x)[0][:, -1]
        return self.head(h)


def dataset(split):
    with np.load(OUTPUT / split / "clips.npz") as data:
        return TensorDataset(torch.from_numpy(data["x"]), torch.from_numpy(data["y"]))


def train():
    torch.set_num_threads(2)
    train_data, test_data = dataset("train"), dataset("test")
    weights = train_data.tensors[1].bincount(minlength=15).float().rsqrt()
    for kind in ("LSTM", "GRU", "TCN"):
        torch.manual_seed(42)
        model = Classifier(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        loss = nn.CrossEntropyLoss(weight=weights)
        print(kind)
        for epoch in tqdm(range(epochs)):
            for x, y in DataLoader(train_data, batch_size=128, shuffle=True):
                optimizer.zero_grad()
                loss(model(x), y).backward()
                optimizer.step()
        model.eval()
        torch.save(model.state_dict(), ROOT / "checkpoints" / f"{kind}.pt")
        with torch.no_grad():
            pred = torch.cat([model(x).argmax(1) for x, _ in DataLoader(test_data, batch_size=128)])
            y = test_data.tensors[1]
            recall = [(pred[y == k] == k).float().mean().item() for k in range(15)]
            print(kind, "accuracy", (pred == y).float().mean().item(), "macro recall", np.mean(recall), flush=True)


if __name__ == "__main__":
    train()
