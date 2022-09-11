from matplotlib.pyplot import bar_label
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
    'batch_size': 64,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 20,
    'epochs': 2
}
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

worker_list = []


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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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


class Worker():
    def __init__(self, name: str, model: Net()) -> None:
        self.name = sy.VirtualWorker(hook=hook, id=name)
        self.model = copy.deepcopy(model).send(self.name)
        self.opt = optim.Adam(params=self.model.parameters(), lr=args['lr'])
        self.id = name

    def train(self, data, target):
        self.opt.zero_grad()
        output = self.model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        loss.backward()
        self.opt.step()
        # print(loss)
        a = loss.get()
        # print(a)
        return(a)


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
    datasets.CIFAR10(
        root='/cifar10',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])]
        )),
    batch_size=args['batch_size'],
    shuffle=True
)


test_loader = DataLoader(
    datasets.CIFAR10(
        '/cifar10',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])]
        )),
    batch_size=args['batch_size'],
    shuffle=False
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
        if batch_idx > 400:
            break
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
                    loss.item()))


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
        model = source.model
        param = model.state_dict()
        for var in param:
            out_param[var] = (param[var] + out_param[var])/2
    # out.load_state_dict(out_param)
    # torch.save(out.state_dict(), 'cifar.pt')
    return(out_param)


@Decorator.timer
def test(workers: list, device, test_loader):
    criterion = nn.CrossEntropyLoss()
    for worker in workers:
        worker.model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确率
    with torch.no_grad():
        for worker in workers:
            worker.model.get()

        # 这里需要
        # final_model_param = combine_pro(workers, final_model)

        final_model_param = combine(
            workers[0].model, workers[1].model, final_model)
        final_model.load_state_dict(final_model_param)

        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            # 将损失加起来
            test_loss += criterion(output, target)
            # 进行预测最可能的分类
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)

            # if index % args['log_interval'] == 0:
            #     print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #         test_loss, correct, len(test_loader.dataset),
            #         100. * correct / len(test_loader.dataset)))
        print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        for worker in workers:
            worker.model.load_state_dict(final_model_param)
            worker.model.send(worker.name)
        # 重点就在这里，之前get了的时候模型就不在远处了，现在要把它送回去


@Decorator.timer
def test_2(workers: list, device, test_loader):
    criterion = nn.CrossEntropyLoss()
    for worker in workers:
        worker.model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确率
    with torch.no_grad():
        for worker in workers:
            worker.model.get()

        # 这里需要
        final_model = workers[0].model
        final_model_param = combine_pro(workers, final_model)

        # final_model_param = combine(
        #     workers[0].model, workers[1].model, final_model)
        final_model.load_state_dict(final_model_param)
        # final_model = workers[0].model
        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            # 将损失加起来
            test_loss += criterion(output, target)
            # 进行预测最可能的分类
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)

            # if index % args['log_interval'] == 0:
            #     print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #         test_loss, correct, len(test_loader.dataset),
            #         100. * correct / len(test_loader.dataset)))
        print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        for worker in workers:
            worker.model.load_state_dict(final_model_param)
            worker.model.send(worker.name)
        # 重点就在这里，之前get了的时候模型就不在远处了，现在要把它送回去


model = Net().to(device)
final_model = copy.deepcopy(model)

qin = Worker('qin', model)
zheng = Worker('zheng', model)
worker_list.append(qin)
worker_list.append(zheng)


@Decorator.timer
def do():
    for epoch in range(1, args['epochs']+1):
        train(args, worker_list, device, federated_train_loader, epoch)
        test_2(worker_list, device, test_loader)


do()
# 问题应该在于copy的问题！要不然怎么会出现少实例化另外一个就解决了的？
