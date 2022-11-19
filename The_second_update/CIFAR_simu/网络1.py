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


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ResNet().to(device)
path = 'cifar10_state.pth'
if os.path.exists(path):
    model.load_state_dict(torch.load(path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(model, data_loader, optimizer, epoch):

    model.train()
    for batch_idx, (x, label) in enumerate(data_loader):
        # if batch_idx > 500:
        #     break
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        Loss_list.append(loss)
        optimizer.step()
        if batch_idx % 64 == 0:
            print('Train Epoch:{}[{}/{}({:.06f}%)]\t  Loss:{:.06f}'.format(
                epoch,
                batch_idx * 64,
                len(train_loader)*64,
                100.*batch_idx/len(train_loader),
                loss
            ))
    return(loss.item())


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for index, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output = model(x)
            test_loss += criterion(output, label)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            test_loss /= len(test_loader.dataset)
            # if index % 40 == 0:
            #     print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #         test_loss, correct, len(test_loader.dataset),
            #         100. * correct / len(test_loader.dataset)))

        return(test_loss, correct * 100)


epochs = 10


Loss_list = []
Acc = []
for epoch in range(epochs):
    print(epoch)
    train_loss = train(model, train_loader, optimizer, epoch)
    _, total_correct = test(model, test_loader)

    print('train loss is: {}'.format(train_loss))
    print('total correct is {}'.format(total_correct))
    # Acc.append(total_correct)
    # x = list(range(1, len(Loss_list)+1))
    # plt.plot(x, Loss_list)
    # plt.show()
