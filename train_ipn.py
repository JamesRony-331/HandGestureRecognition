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
                # f(x)=max(0,x)
                # in_channels：输入通道数量。
                # out_channels = 64：卷积输出64 个特征通道。
                # kernel_size = 3：每次卷积使用三个时间位置。
                # dilation：膨胀系数，控制卷积核取样的间隔。扩大卷积层可以观察到的时间范围。
                # padding = dilation：在当前配置下保持时间长度不变。
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = getattr(nn, kind)(63, 64, batch_first=True)
            #单层、单向 LSTM 或 GRU。
        self.head = nn.Linear(64, 15)

    def forward(self, x):
        if self.kind == "TCN":
            h = self.encoder(x.transpose(1, 2)).mean(2)
        else:
            h = self.encoder(x)[0][:, -1]
        return self.head(h)


def dataset(split):#读取保存的数据集
    with np.load(OUTPUT / split / "clips.npz") as data:
        return TensorDataset(torch.from_numpy(data["x"]), torch.from_numpy(data["y"]))
    # x手势动作的关键点序列(5000, 32, 63)
    # y 每个动作的类别编号(5000, )
    #TensorDataset 的作用是把输入数据与对应标签组合成一个数据集。


def train():
    torch.set_num_threads(2)#限制 CPU 线程数量
    train_data, test_data = dataset("train"), dataset("test")
    weights = (train_data.tensors[1]
               .bincount(minlength=15)#14个类别出现的次数
               .float().rsqrt())#计算类别权重
    #wk=1/sqrt(nk) 其中 nk 是类别 k 的样本数量。样本数量越少，权重越大。
    for kind in ("LSTM", "GRU", "TCN"):
        torch.manual_seed(42)#固定随机种子
        model = Classifier(kind)#创建模型
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)#使用 Adam 优化算法。
        loss = nn.CrossEntropyLoss(weight=weights) #交叉熵损失函数（Cross-Entropy Loss），
        #y=[ylogy^+(1-y)log(1-y^)]
        print(kind)
        for epoch in tqdm(range(epochs)):
            for x, y in DataLoader(train_data, batch_size=128, shuffle=True): #分批读取数据
                optimizer.zero_grad()#清除上一次计算的梯度
                loss(model(x), y).backward()
                optimizer.step()#更新
        model.eval()
        torch.save(model.state_dict(), ROOT / "checkpoints" / f"{kind}.pt")
        with torch.no_grad():
            pred = torch.cat([model(x).argmax(1) for x, _ in DataLoader(test_data, batch_size=128)])
            y = test_data.tensors[1]
            recall = [(pred[y == k] == k).float().mean().item() for k in range(15)]
            print(kind, "accuracy", (pred == y).float().mean().item(), "macro recall", np.mean(recall), flush=True)


if __name__ == "__main__":
    train()
