import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from fedlab.contrib.dataset import PartitionedCIFAR10

from torch.utils.data import DataLoader
import time
import copy
from threading import Thread
from fun_cifar import *
import matplotlib.pyplot as plt

train_args = {
    'use_cuda': True,
    'num_clients': 16,
    'batch_size': 64,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 40,
    'aggre_interval': 1,
    'epochs': 20,
    'initial_rate_low': 1,
    'initial_rate_high': 100,
    'rate_change_low': -10,
    'rate_change_high': 10,
    'recv_threshold': 1.9444,
    'wait_threshold': 1.9444
}
use_cuda = train_args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
UE_list = []

# Data distribute


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


# Class for local user equipment


class UE():
    def __init__(self, name: str, model: Net()) -> None:
        self.id = name
        self.model = copy.deepcopy(model)
        self.opt = optim.Adam(
            params=self.model.parameters(), lr=train_args['lr'])
        self.channel_rate = train_args['initial_rate_low'] + \
            (train_args['initial_rate_high'] -
             train_args['initial_rate_low'])*np.random.rand()  # 每个ue的信道速率,随机初始化
        self.trans_delay = 0  # 传输耗时
        self.label = 0

    def train(self, data, target):  # Local training of single device
        self.opt.zero_grad()
        output = self.model(data)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        loss.backward()
        self.opt.step()
        return(loss.item())


# Initialize UE, num for UE number
def init_ue(num: int):
    for i in range(num):
        user = UE('user'+str(i+1), model)
        UE_list.append(user)


def aggregate_with_base_value(UEs: list):
    out_param = UEs[0].model.state_dict()
    for var in out_param:
        out_param[var] = 0

    aggre_list = BS_receive(UEs, train_args, True)
    for ue in aggre_list:
        if ue.label != 0:
            print('{}, label: {}'.format(ue.id, ue.label))
            param = ue.model.state_dict()
            for var in param:  # 这里又将param的元素转成gpu上了
                out_param[var] = param[var].cuda() + out_param[var]
    for var in out_param:
        out_param[var] = out_param[var]/(len(UEs))
    return(out_param)


# def aggregate_with_base_value(UEs: list):
#     out_param = UEs[0].model.state_dict()
#     for var in out_param:
#         out_param[var] = 0
#     for ue in UEs:
#         param = ue.model.state_dict()
#         Param_compression(param, 2)
#         for var in param:  # 这里又将param的元素转成gpu上了
#             out_param[var] = param[var].cuda() + out_param[var]

#     for var in out_param:
#         out_param[var] = out_param[var]/(len(UEs))

#     return(out_param)


# Model aggregate with full value
def aggregate_with_full_model(UEs: list):
    out_param = UEs[0].model.state_dict()
    for var in out_param:
        out_param[var] = 0
    for ue in UEs:
        param = ue.model.state_dict()
        for var in param:  # 这里又将param的元素转成gpu上了
            out_param[var] = param[var].cuda() + out_param[var]

    for var in out_param:
        out_param[var] = out_param[var]/(len(UEs))

    return(out_param)


# Load data for test
test_loader = DataLoader(
    datasets.CIFAR10(
        '/cifar10',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])]
        )),
    batch_size=train_args['batch_size'],
    shuffle=False
)

hetero = PartitionedCIFAR10(
    root='/cifar10',
    path='/cifar10_hetero_dir.pkl',
    dataname="cifar10",
    num_clients=train_args['num_clients'],
    download=False,
    preprocess=False,
    balance=False,
    partition="dirichlet",
    seed=2022,
    dir_alpha=0.3,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])]
    ),
    target_transform=transforms.ToTensor()
)


def train(train_args, UEs: list, device, epoch):
    for ue in UEs:
        ue.model.train()

    for id, ue in enumerate(UE_list):
        train_loader = hetero.get_dataloader(
            id, batch_size=train_args['batch_size'])
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx > 100:
            #     break
            ue_data = data.to(device)
            ue_target = target.to(device)
            loss = ue.train(ue_data, ue_target)  # Loss from distant UE


def test(UEs: list, device, test_loader, **kw):
    for ue in UEs:
        ue.model.eval()
    test_loss = 0
    correct = 0  # correct rate
    with torch.no_grad():
        final_model_param = UEs[0].model.state_dict()
        if kw['method'] == 'base':
            final_model_param = aggregate_with_base_value(UEs)
        elif kw['method'] == 'full':
            final_model_param = aggregate_with_full_model(UEs)
        BS_model.load_state_dict(final_model_param)

        for ue in UEs:
            ue.model.load_state_dict(final_model_param)

        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = BS_model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            test_loss /= len(test_loader.dataset)
        return(test_loss, correct / 100)


model = Net().to(device)
BS_model = copy.deepcopy(model)

init_ue(16)
t = Thread(target=Channel_rate, args=(UE_list, train_args,), daemon=True)
t.start()
set_method(train_args, 'good')


total_correct_list = []
for epoch in range(1, train_args['epochs']+1):
    train(train_args, UE_list, device, epoch)
    _, total_correct = test(UE_list, device, test_loader, method='base')
    print('total correct is {}'.format(total_correct))
    total_correct_list.append(total_correct)

x = list(range(1, len(total_correct_list)+1))
print(total_correct_list)
plt.plot(x, total_correct_list)
plt.xlabel('epoch')
plt.ylabel('accuracy/%')
plt.xticks(np.arange(0, 22, 2))
plt.grid(linestyle='--')
plt.show()
