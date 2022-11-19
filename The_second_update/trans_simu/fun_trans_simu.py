from collections import OrderedDict
import torch
import time
import numpy as np


def Quant(Vx, Q, RQM):
    return round(Q * Vx) - RQM


def QuantRevert(VxQuant, Q, RQM):
    return (VxQuant + RQM) / Q


def ListQuant(data_list, quant_bits):
    # 数组范围估计
    data_min = min(data_list)
    data_max = max(data_list)

    # 量化参数估计
    Q = ((1 << quant_bits) - 1) * 1.0 / (data_max - data_min)
    RQM = (int)(np.round(Q*data_min))

    # 产生量化后的数组
    quant_data_list = []
    for x in data_list:
        quant_data = Quant(x, Q, RQM)
        quant_data_list.append(quant_data)
    quant_data_list = np.array(quant_data_list)
    return (Q, RQM, quant_data_list)


def ListQuantRevert(quant_data_list, Q, RQM):
    quant_revert_data_list = []
    for quant_data in quant_data_list:
        # 量化数据还原为原始浮点数据
        revert_quant_data = QuantRevert(quant_data, Q, RQM)
        quant_revert_data_list.append(revert_quant_data)
    quant_revert_data_list = np.array(quant_revert_data_list)
    return quant_revert_data_list


def Param_compression(dict: OrderedDict, Bits):
    # 输入是model的state_dict,原地进行操作，输出是模型大小(bit)
    size = 0
    for key, value in dict.items():
        temp = value.cpu().numpy()  # 放在cpu上才能转为numpy
        sh = temp.shape
        temp = temp.reshape(1, -1)
        Q, RQM, temp[0] = ListQuant(temp[0], Bits)
        temp[0] = ListQuantRevert(temp[0], Q, RQM)
        size += (4+Bits)*np.size(temp)
        temp = torch.from_numpy(temp.reshape(sh))
        dict[key] = temp
    return(10*size/(1024*1024)+np.random.randint(-2, 2))


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


def set_method(args: dict, quality: str):
    if quality == 'perfect':
        args['recv_threshold'] = 11.472
    elif quality == 'good':
        args['recv_threshold'] = 2.82
        args['wait_threshold'] = 2.611
    elif quality == 'mid':
        args['recv_threshold'] = 1.8
        args['wait_threshold'] = 1.88
    elif quality == 'bad':
        args['recv_threshold'] = 1.41
        args['wait_threshold'] = 1.468
    else:
        exit()
    return(args)


def calculte_size(dict: OrderedDict, Bits):
    size = 0
    for key, value in dict.items():
        temp = value.cpu().numpy()  # 放在cpu上才能转为numpy
        sh = temp.shape
        temp = temp.reshape(1, -1)
        size += (4+Bits)*np.size(temp)
        temp = torch.from_numpy(temp.reshape(sh))
        dict[key] = temp
    return(10*size/(1024*1024))


def BS_receive(UEs: list, train_args, grading):
    aggre_list = UEs  # 最后将要进行本轮模型聚合的列表
    for ue in UEs:
        param = ue.model.state_dict()
        data_size = calculte_size(param, 2)
        ue.trans_delay = Trans_delay(ue, data_size)  # 计算完这个ue的model大小
        if ue.trans_delay > train_args['recv_threshold']:
            ue.label = 0  # 直接不参与后面的计算，但不能直接剔除
        elif grading == True:  # 基础值已经收到了该接受增量值了
            data_size = calculte_size(param, 4)
            ue.trans_delay = Trans_delay(ue, data_size)
            if ue.trans_delay < train_args['wait_threshold']:
                Param_compression(param, 4)
                ue.model.load_state_dict(param)
                ue.label = 4
            else:
                Param_compression(param, 2)
                ue.model.load_state_dict(param)
                ue.label = 2
        else:
            break
    return(aggre_list)
