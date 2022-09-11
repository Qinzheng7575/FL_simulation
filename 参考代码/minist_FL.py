import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
from torch.utils.data import DataLoader
import time
import copy
hook = sy.TorchHook(torch)

args = {
    'use_cuda': True,
    'batch_size': 16,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 40,
    'epochs': 2
}
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

worker_list = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            # 输出26*26*32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1),
            # 输出24*24*64
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )
        self.dropout = nn.Dropout2d(0.25)  # 随机丢弃

    def forward(self, x):
        x = self.conv(x)  # 输入的时候是28*28*1,输出应该是24*24*64
        x = F.max_pool2d(x, 2)  # 用步长为2的池化,输出12*12*64
        x = x.view(-1, 64*12*12)  # 此时将其拉成一条直线进入全连接
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class Worker():
    def __init__(self, name: str, model: Net()) -> None:
        self.name = sy.VirtualWorker(hook=hook, id=name)
        self.model = copy.deepcopy(model).send(self.name)
        self.opt = optim.SGD(params=self.model.parameters(), lr=args['lr'])
        self.id = name

    def train(self, data, target):
        self.opt.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.opt.step()
        return(loss.get())


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
        return wrapper


federated_train_loader = DataLoader(
    datasets.MNIST(
        root='/minist_data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )),
    batch_size=args['batch_size'],
    shuffle=True
)


test_loader = DataLoader(
    datasets.MNIST(
        '/minist_data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=args['batch_size'],
    shuffle=True
)


def alloc(data: list, num: int, rank: int):
    block = int(len(data)/num)
    begin = rank
    end = rank + block
    return(data[begin:end])


@Decorator.timer
def train(args, workers: list, device, train_loader, epoch):
    for worker in workers:
        worker.model.train()
    # 远程迭代
    for batch_idx, (data, target) in enumerate(train_loader):
        # minist下降的实在是太快了，不过如果传输时间也在同一个程度的话，
        # 其实也不是问题，另一个担心的是卷积网络模型太小。
        # if batch_idx > 50:
        #     break

        for rank, worker in enumerate(workers):
            # 分发数据
            data_alloc = alloc(data, len(workers), rank)
            target_alloc = alloc(target, len(workers), rank)
            worker_data = data_alloc.send(worker.name).to(device)
            worker_target = target_alloc.send(worker.name).to(device)

        # 训练
            loss = worker.train(worker_data, worker_target)
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch:{}[{}/{}({:.06f}%)]\t {} Loss:{:.06f}'.format(
                    epoch,
                    batch_idx * args['batch_size'],
                    len(train_loader)*args['batch_size'],
                    100.*batch_idx/len(train_loader),
                    worker.id,
                    loss.item()
                )
                )


def combine(a: Net, b: Net, c: Net):
    a_param, b_param, c_param = a.state_dict(), b.state_dict(), c.state_dict()
    for var in c_param:
        # print(var)
        c_param[var] = (a_param[var] + b_param[var])/2
    c.load_state_dict(c_param, strict=True)
    return(c_param)
    # print('combine ok')

# 为以后多个设备聚合做准备


def combine_pro(source_list: list, out: Net):
    out_param = out.state_dict()
    for source in source_list:
        param = source.state_dict()
        for var in param:
            out_param[var] = (param[var] + out_param[var])/2
    out.load_state_dict(out_param)
    return(out_param)


@Decorator.timer
def test(workers: list, device, test_loader):
    for worker in workers:
        worker.model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确率
    with torch.no_grad():
        for worker in workers:
            worker.model.get()

        # 这里需要

        final_model_param = combine(
            workers[0].model, workers[1].model, final_model)

        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            # 将损失加起来
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 进行预测最可能的分类
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)

            if index % args['log_interval'] == 0:
                print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

        for worker in workers:
            worker.model.load_state_dict(final_model_param)
            worker.model.send(worker.name)
            # 重点就在这里，之前get了的时候模型就不在远处了，现在要把它送回去


model = Net().to(device)
final_model = copy.deepcopy(model)

user1 = Worker('user1', model)
user2 = Worker('user2', model)
user3 = Worker('user3', model)
user4 = Worker('user4', model)


worker_list.append(user1)
worker_list.append(user2)
worker_list.append(user3)
worker_list.append(user4)


@Decorator.timer
def do():
    for epoch in range(1, args['epochs']+1):
        train(args, worker_list, device, federated_train_loader, epoch)
        test(worker_list, device, test_loader)


do()
