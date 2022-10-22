import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
from torch.utils.data import DataLoader
import time
import copy
from functions_for_trans import *
hook = sy.TorchHook(torch)
train_args = {
    'use_cuda': True,
    'batch_size': 16,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 40,
    'aggre_interval': 1,
    'epochs': 1
}
use_cuda = train_args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
UE_list = []


# Timer
class Decorator:
    def __init__(self):
        pass

    @staticmethod
    def timer(func):
        def wrapper(*args, **kw):
            start = time.time()
            func(*args, **kw)
            last = time.time()-start
            print("\n----Function{} cost time :{:.3f}----".format(func.__name__, last))
        return wrapper


# Data distribute
def distribute(data: list, num: int, rank: int):
    block = int(len(data)/num)
    begin = rank
    end = rank + block
    return(data[begin:end])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*12*12)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# Class for local user equipment
class UE():
    def __init__(self, name: str, model: Net()) -> None:
        self.name = sy.VirtualWorker(hook=hook, id=name)  # tag for pysyft
        self.id = name
        self.model = copy.deepcopy(model).send(self.name)
        self.opt = optim.SGD(
            params=self.model.parameters(), lr=train_args['lr'])

    def train(self, data, target):  # Local training of single device
        self.opt.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.opt.step()
        return(loss.get())


# Initialize UE, num for UE number
def init_ue(num: int):
    for i in range(num):
        user = UE('user'+str(i+1), model)
        UE_list.append(user)


# Model aggregate with base value
@Decorator.timer
def aggregate_with_base_value(source_list: list, out: Net):
    out_param = out.state_dict()
    for source in source_list:
        param = source.model.state_dict()
        Param_compression(param)
        Param_recovery(param)
        for var in param:
            out_param[var] = (param[var].to(device) + out_param[var])/2
    out.load_state_dict(out_param)
    return(out_param)


# Model aggregate with full value
@Decorator.timer
def aggregate_with_full_model(source_list: list, out: Net):
    out_param = out.state_dict()
    for source in source_list:
        param = source.model.state_dict()
        for var in param:
            out_param[var] = (param[var] + out_param[var])/2
    out.load_state_dict(out_param)
    return(out_param)


# Load data for training
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

# Load data for test
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


# FL system train
@Decorator.timer
def train(train_args, UEs: list, device, train_loader, epoch):
    for ue in UEs:
        ue.model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx > 100:
        #     break
        for rank, ue in enumerate(UEs):
            # distribute data
            data_distribute = distribute(data, len(UEs), rank)
            target_distribute = distribute(target, len(UEs), rank)
            ue_data = data_distribute.send(ue.name).to(device)
            ue_target = target_distribute.send(ue.name).to(device)
            # training
            loss = ue.train(ue_data, ue_target)  # Loss from distant UE
            if batch_idx % train_args['log_interval'] == 0:
                print('Train Epoch:{}[{}/{}({:.06f}%)]\t {} Loss:{:.06f}'.format(
                    epoch,
                    batch_idx * train_args['batch_size'],
                    len(train_loader)*train_args['batch_size'],
                    100.*batch_idx/len(train_loader),
                    ue.id,
                    loss.item()
                ))
                Loss_list.append(loss.item())


# BS performs aggregation and testing
def test(UEs: list, device, test_loader, **kw):
    for ue in UEs:
        ue.model.eval()
    test_loss = 0
    correct = 0  # correct rate
    with torch.no_grad():
        for ue in UEs:
            ue.model.get()

        if kw['method'] == 'base':
            final_model_param = aggregate_with_base_value(UEs, BS_model)
        elif kw['method'] == 'full':
            final_model_param = aggregate_with_full_model(UEs, BS_model)

        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = BS_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
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


model = Net().to(device)
BS_model = copy.deepcopy(model)
Loss_list = []
init_ue(4)
for epoch in range(1, train_args['epochs']+1):
    # train(train_args, UE_list, device, federated_train_loader, epoch)

    # Select the model transport method according to the 'method'
    # test(UE_list, device, test_loader, method='base')
    # test(UE_list, device, test_loader, method='full')
