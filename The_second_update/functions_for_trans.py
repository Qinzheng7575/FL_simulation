from collections import OrderedDict
import torch
import time
import numpy as np


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
        temp = value.cpu().numpy()  # 放在cpu上才能转为numpy
        sh = temp.shape
        temp = temp.reshape(1, -1)
        for i, num in enumerate(temp[0]):
            temp[0][i] = Quantify(num, Bits)
        size += (4+Bits)*np.size(temp)
        temp = torch.from_numpy(temp.reshape(sh))
        # print(temp)
        dict[key] = temp
    return(size)

# model = Net()
# model_param = model.state_dict()
# size0 = Param_compression(model_param, 4)
# print(size0)


def Channel_rate(UE_list):
    # 作为一个独立的后台线程，职责是为主线程提供信道速度
    while True:
        for ue in UE_list:
            ue.channel_rate += np.random.rand()
        time.sleep(2)


def Trans_delay(ue, size):
    delay = size/ue.channel_rate
    return(delay)


'''How they used?
t = Thread(target=Channel_rate, args=(UE_list,), daemon=True)
t.start()
'''
