# 请先阅读readme.md
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
train_args = {
    'use_cuda': True,
    'batch_size': 16,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 40,
    'aggre_interval': 1,
    'epochs': 2
}
use_cuda = train_args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
UE_list = []


# 计时器
class Decorator:
    def __init__(self):
        pass

    @staticmethod
    def timer(func):
        def wrapper(*args, **kw):
            start = time.time()
            func(*args, **kw)
            last = time.time()-start
            print("\n----函数{}花费时间{:.3f}----".format(func.__name__, last))
        return wrapper


# 分发数据
def alloc(data: list, num: int, rank: int):
    block = int(len(data)/num)
    begin = rank
    end = rank + block
    return(data[begin:end])


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


# 本地用户类
class UE():
    def __init__(self, name: str, model: Net()) -> None:
        self.name = sy.VirtualWorker(hook=hook, id=name)  # 用于pysyft识别的标记
        self.id = name  # 用于打印输出的名字
        self.model = copy.deepcopy(model).send(self.name)
        self.opt = optim.SGD(
            params=self.model.parameters(), lr=train_args['lr'])

    def train(self, data, target):  # 单个设备进行本地训练，训练的数据和model都存储在本地
        self.opt.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.opt.step()
        return(loss.get())


@Decorator.timer
def transport():  # 模拟传输过程，
    return(0)


# 模型聚合
def aggregate(source_list: list, out: Net):
    out_param = out.state_dict()
    for source in source_list:
        param = source.model.state_dict()
        for var in param:
            out_param[var] = (param[var] + out_param[var])/2
    out.load_state_dict(out_param)
    return(out_param)


# 训练数据加载
federated_train_loader = DataLoader(
    datasets.MNIST(
        root='/minist_data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )),
    batch_size=train_args['batch_size'],
    shuffle=True
)

# 测试数据加载
test_loader = DataLoader(
    datasets.MNIST(
        '/minist_data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=train_args['batch_size'],
    shuffle=True
)


# 执行本地模型训练
def train(train_args, UEs: list, device, train_loader, epoch):

    for ue in UEs:
        ue.model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx > 100:
        #     break
        for rank, ue in enumerate(UEs):
            # 分发数据
            data_alloc = alloc(data, len(UEs), rank)
            target_alloc = alloc(target, len(UEs), rank)
            ue_data = data_alloc.send(ue.name).to(device)
            ue_target = target_alloc.send(ue.name).to(device)
            # 训练
            loss = ue.train(ue_data, ue_target)  # 此loss是从远处UE提取过来的
            if batch_idx % train_args['log_interval'] == 0:
                print('Train Epoch:{}[{}/{}({:.06f}%)]\t {} Loss:{:.06f}'.format(
                    epoch,
                    batch_idx * train_args['batch_size'],
                    len(train_loader)*train_args['batch_size'],
                    100.*batch_idx/len(train_loader),
                    ue.id,
                    loss.item()
                )
                )


# BS端进行聚合、测试
def test(UEs: list, device, test_loader, epoch):
    for ue in UEs:
        ue.model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确率
    with torch.no_grad():
        for ue in UEs:
            ue.model.get()

        final_model_param = aggregate(
            UEs, BS_model)

        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = BS_model(data)
            # 将损失加起来
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 进行预测最可能的分类
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)

            if index % train_args['log_interval'] == 0:
                print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

        for ue in UEs:
            ue.model.load_state_dict(final_model_param)
            ue.model.send(ue.name)
            # 重点就在这里，之前get了的时候模型就不在远处了，现在要把它送回去


# 初始化UE,num为个数
def init_ue(num: int):
    for i in range(num):
        user = UE('user'+str(num), model)
        UE_list.append(user)


model = Net().to(device)
BS_model = copy.deepcopy(model)

init_ue(4)

for epoch in range(1, train_args['epochs']+1):
    train(train_args, UE_list, device, federated_train_loader, epoch)

    # 模拟传输过程
    if epoch % train_args['aggre_interval'] == 0:
        transport()

    test(UE_list, device, test_loader, epoch)
