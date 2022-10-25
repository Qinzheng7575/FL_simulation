from functions_for_trans import *
from collections import OrderedDict
import time
from threading import Thread
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import syft as sy
import torch.optim as optim
import copy
hook = sy.TorchHook(torch)
# torch.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=math.inf)
train_args = {
    'use_cuda': True,
    'batch_size': 16,
    'test_batch_size': 2000,
    'lr': 0.01,
    'log_interval': 40,
    'aggre_interval': 1,
    'epochs': 1
}

c = torch.tensor([[[[-0.2771,  0.2636,  0.2257],
                  [-0.2511,  0.1435,  0.2170],
    [0.3056, -0.2653, -0.1759]]],


    [[[0.2036,  0.3149, -0.2843],
      [-0.3170, -0.0894,  0.1554],
      [0.0864,  0.0028,  0.1756]]],


    [[[0.2038,  0.1557,  0.1535],
      [0.1720, -0.1163, -0.2585],
      [0.3092, -0.0816, -0.2654]]],


    [[[-0.1752, -0.1801,  0.0392],
      [0.0701, -0.0612, -0.0441],
      [0.2253, -0.2680,  0.1270]]],


    [[[0.0852,  0.0536, -0.0018],
      [-0.0827, -0.2699,  0.2513],
      [0.1718, -0.0571,  0.1373]]],


    [[[-0.2258,  0.3267,  0.1291],
      [0.3090,  0.2631,  0.0960],
      [-0.0949,  0.2459, -0.0846]]],


    [[[-0.2203,  0.0783,  0.1618],
      [-0.1296, -0.1403, -0.2414],
      [0.2958, -0.1453,  0.2728]]],


    [[[0.1636, -0.0231,  0.1646],
      [-0.3075,  0.2151,  0.2970],
      [0.3179,  0.0620, -0.3311]]],


    [[[0.1352, -0.2968, -0.0960],
      [-0.1356, -0.3329,  0.0179],
      [-0.1962,  0.0226,  0.2846]]],


    [[[0.0483,  0.2408, -0.2706],
      [-0.0154, -0.0773, -0.1957],
      [0.0154,  0.0908,  0.2842]]],


    [[[0.0210,  0.0317, -0.2694],
      [0.1694, -0.3239, -0.0995],
      [0.2944,  0.3269,  0.0872]]],


    [[[0.1050, -0.2720, -0.1734],
      [-0.1106,  0.2375,  0.3273],
      [0.0249,  0.1635,  0.1797]]],


    [[[0.1495,  0.1488,  0.0265],
      [-0.2211,  0.1617, -0.2346],
      [-0.1953,  0.0532, -0.1746]]],


    [[[0.3120,  0.0912,  0.0078],
      [0.2333,  0.1538, -0.2504],
      [0.2276, -0.1021,  0.0287]]],


    [[[-0.2097, -0.1901,  0.3187],
      [0.0474, -0.2742,  0.1212],
      [0.3251, -0.2218, -0.0794]]],


    [[[0.0426, -0.0490,  0.0551],
      [0.1432, -0.1073, -0.0281],
      [-0.1697,  0.1731, -0.1905]]],


    [[[-0.2497, -0.1130, -0.2524],
      [-0.1410, -0.3229,  0.1861],
      [0.2898,  0.3007,  0.1119]]],


    [[[0.0299,  0.1403, -0.0959],
      [0.0207,  0.0519,  0.2580],
      [-0.1649,  0.1885,  0.0312]]],


    [[[0.1390, -0.1054,  0.2255],
      [-0.1343, -0.0406, -0.2935],
      [-0.0838,  0.1657, -0.1040]]],


    [[[-0.1993, -0.1654,  0.1394],
      [0.0476, -0.1602,  0.1816],
      [0.2444,  0.1074, -0.1147]]],


    [[[0.2184,  0.2475,  0.1685],
      [-0.1878, -0.1550, -0.1981],
      [-0.0085, -0.0162,  0.1390]]],


    [[[0.2709,  0.2770,  0.2778],
      [-0.1917,  0.0994, -0.0102],
      [-0.2407,  0.2095, -0.0322]]],


    [[[-0.1135,  0.1569, -0.2864],
      [-0.2545, -0.2451, -0.1901],
      [-0.2559,  0.0755, -0.0803]]],


    [[[-0.0899, -0.3124, -0.1809],
      [-0.2075,  0.2634,  0.2461],
      [-0.1238,  0.1975, -0.1452]]],


    [[[-0.1552, -0.0632, -0.0811],
      [-0.1990, -0.3186,  0.1823],
      [-0.0370,  0.0730, -0.1399]]],


    [[[0.2569,  0.1510,  0.3325],
      [-0.0751, -0.2857,  0.0358],
      [0.1179,  0.0438,  0.2478]]],


    [[[0.2349, -0.3177, -0.2572],
      [0.3144, -0.0413, -0.0948],
      [0.3016,  0.3239, -0.0380]]],


    [[[-0.0453,  0.0620,  0.1883],
      [0.2295,  0.2558,  0.3177],
      [0.2896, -0.1812,  0.2725]]],


    [[[0.1820,  0.1817,  0.0368],
      [0.2283, -0.2836,  0.0503],
      [0.2994,  0.0645, -0.1049]]],

    [[[0.3110, -0.2522, -0.1878],
      [-0.0028, -0.2283, -0.0242],
      [0.0861, -0.0432,  0.0616]]],


    [[[0.1158, -0.0384, -0.1739],
      [0.1602,  0.1205,  0.3252],
      [-0.0705,  0.2830, -0.0531]]],


    [[[-0.2113, -0.2368,  0.1091],
      [0.0332,  0.0100, -0.0945],
      [-0.2107, -0.2088,  0.2416]]]])

c1 = c.numpy()


def Quantify(number, Bits):
    if number < 0:
        number = abs(number)
        neg = -1
    else:
        neg = 1

    integer = np.int(number)
    number -= integer
    digit = 0  # 小数点后0的个数
    while 10*number < 1:
        digit += 1
        number *= 10

    precision = 1/pow(2, Bits)
    # print('precision is {}\n'.format(precision))
    quant = 0
    while quant+precision < number:
        quant += precision

    # print('after quantilize is {}\n'.format(quant))
    # print('loss is {}\n'.format(number-quant))
    return(neg*(integer+quant/pow(10, digit)))


def Param_compression(dict: OrderedDict, Bits):
    # 输入是model的state_dict,原地进行操作，输出是模型大小(bit)
    size = 0
    for key, value in dict.items():
        temp = value.numpy()
        sh = temp.shape
        temp = temp.reshape(1, -1)
        for i, num in enumerate(temp[0]):
            temp[0][i] = Quantify(num, Bits)
        size += (4+Bits)*np.size(temp)
        temp = torch.from_numpy(temp.reshape(sh))
        # print(temp)
        dict[key] = temp
    return(size)


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


model = Net()
model_param = model.state_dict()
print(model_param['fc.2.weight'])
size0 = Param_compression(model_param, 4)
print(model_param['fc.2.weight'])


class UE():
    def __init__(self, name: str, model: Net()) -> None:
        self.name = sy.VirtualWorker(hook=hook, id=name)  # tag for pysyft
        self.id = name
        self.model = copy.deepcopy(model).send(self.name)
        self.opt = optim.SGD(
            params=self.model.parameters(), lr=train_args['lr'])
        self.channel_rate = np.random.rand()  # 每个ue的信道速率,随机初始化
        self.trans_delay = 0  # 传输耗时

    def train(self, data, target):  # Local training of single device
        self.opt.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.opt.step()
        return(loss.get())


def Channel_rate(UE_list):
    # 作为一个独立的后台线程，职责是为主线程提供信道速度
    while True:
        for ue in UE_list:
            ue.channel_rate += np.random.rand()
        time.sleep(2)


def Trans_delay(ue: UE, size):
    delay = size/ue.channel_rate
    return(delay)


'''How they used?
t = Thread(target=Channel_rate, args=(UE_list,), daemon=True)
t.start()
'''
