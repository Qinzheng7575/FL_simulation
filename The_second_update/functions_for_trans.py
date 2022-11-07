from collections import OrderedDict
import torch
import time
import numpy as np
import copy


def Quantify(number, Bits):  # number是0的时候，卡死了
    if number == 0:
        return(0)
    elif number < 0:
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
        dict[key] = temp
    return(10*size/(1024*1024)+np.random.randint(-5, 5))


def Channel_rate(UE_list, train_args):
    # 作为一个独立的后台线程，职责是为主线程提供信道速度
    while True:
        for ue in UE_list:
            ue.channel_rate += train_args['rate_change_low'] +\
                (train_args['rate_change_high'] -
                 train_args['rate_change_low'])*np.random.rand()
        time.sleep(5)


def Trans_delay(ue, size):
    delay = size/ue.channel_rate
    return(delay)


'''How they used?
t = Thread(target=Channel_rate, args=(UE_list,), daemon=True)
t.start()
'''


def BS_receive(UEs: list, train_args, grading):
    aggre_list = UEs  # 最后将要进行本轮模型聚合的列表
    for ue in UEs:  # 先遍历一遍进行压缩并计算时延
        param = copy.deepcopy(ue.model.state_dict())  # 复制
        data_size = Param_compression(param, 2)
        ue.model.load_state_dict(param)
        ue.trans_delay = Trans_delay(ue, data_size)

    for ue in UEs:
        if ue.trans_delay > train_args['recv_threshold']:
            aggre_list.remove(ue)  # 直接不接收
        elif grading == True:  # 基础值已经收到了该接受补充值了
            param = copy.deepcopy(ue.model.state_dict())
            data_size = Param_compression(param, 4)
            ue.trans_delay = Trans_delay(ue, data_size)
            if ue.trans_delay < train_args['wait_threshold']:
                ue.model.load_state_dict(param)  # 在等待时间以内，才能使用补充值的模型
            else:
                break
        else:
            break
    return(aggre_list)
