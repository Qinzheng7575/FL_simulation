import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
# 设置中文显示
font = {'family': 'Microsoft YaHei', 'size': 12}
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', **font)


class Decorator:
    def __init__(self):
        pass

    @staticmethod
    def timer(func):  # 计时器
        def wrapper(*args, **kw):
            start = time.time()
            func(*args, **kw)
            last = time.time()-start
            print("\n----函数{}花费时间{:.3f}----".format(func.__name__, last))
            with open("acc.txt", "a") as f:
                f.write("{:.3f},".format(last))
                f.close()
        return wrapper


batch_size = 64
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[
                         0.2008, 0.2008, 0.2008])
])

train_data = datasets.CIFAR10(
    root='/cifar10',
    train=True,
    download=False,
    transform=transform
)

test_data = datasets.CIFAR10(
    root='/cifar10',
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        self.block1 = ResBlock(64, 128, stride=2)
        self.block2 = ResBlock(128, 256, stride=2)
        self.block3 = ResBlock(256, 512, stride=2)
        self.block4 = ResBlock(512, 512, stride=2)
        self.fc = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


model = ResNet().to(device)
path = 'cifar10_state.pth'
if os.path.exists(path):
    model.load_state_dict(torch.load(path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(model, data_loader, optimizer, epoch):

    model.train()
    for batch_idx, (x, label) in enumerate(data_loader):
        # if batch_idx > 400:
        #     break
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # if batch_idx % 20 == 0:
        #     print('epoch', epoch)
        #     print(batch_idx, 'loss:', loss.item())
    return(loss.item())


def test(model, test_loader):
    model.eval()
    total_correct = 0.
    test_loss = 0.
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            output = model(x)
            test_loss += criterion(output, label)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(label).sum().item()

        total_size = len(test_loader.dataset)
        test_loss /= total_size
        total_correct /= total_size
        # print('test loss: ', test_loss)
        # print('acc: ', total_correct * 100)
        return(test_loss, total_correct * 100)


epochs = 20
# model.load_state_dict(torch.load('cifar.pt'))

Loss_list = []
Acc = []
for epoch in range(epochs):
    print(epoch)
    train_loss = train(model, train_loader, optimizer, epoch)
    _, total_correct = test(model, test_loader)

    # print('train loss is: {}'.format(train_loss))
    Loss_list.append(train_loss)
    Acc.append(total_correct)

with open("acc1.txt", "a") as f:
    f.write(str(Acc))

    f.close()
with open("loss1.txt", "a") as f:
    f.write(str(Loss_list))

    f.close()


x = range(1, len(Loss_list)+1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
h1 = ax.plot(x, Loss_list, '.-', color='r', label='Loss')
ax.set_ylabel('Training loss')
ax.set_xlabel('epoch')


ax2 = ax.twinx()
h2, = ax2.plot(x, Acc, 'x--', color='b', label='accuracy')
ax2.set_ylabel('Accuracy of test')


ax2.spines['left'].set_color('r')
ax2.spines['right'].set_color('b')
ax.tick_params(axis='y', colors='r')
ax2.tick_params(axis='y', colors='b')
ax.yaxis.label.set_color('r')
ax2.yaxis.label.set_color('b')

ax.legend(loc=6)
ax2.legend(loc=7)
plt.xticks(np.arange(0, 22, 2))
plt.grid(linestyle='--')
plt.show()
