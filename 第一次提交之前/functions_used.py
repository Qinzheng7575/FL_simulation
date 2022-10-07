from collections import OrderedDict
import torch
import numpy as np


def float_to_str(num: float):  # 将float科学计数法后转化为字符串
    d = str(num)
    if d[0] == '-':
        neg = True
    else:
        neg = False
    point_pos = d.find('.')
    neednt = False if abs(num) > 1 or 'e' in d else True  # 不需要科学计数法了

    for i in range(len(d)-point_pos):
        if d[i+point_pos] > '0':
            num_pos = i+point_pos
            break
    if not neednt:
        return(d)
    else:
        d = d[num_pos:num_pos+1]+'.' + \
            d[num_pos+1:]+'e-'+str(num_pos-point_pos)
        if neg:
            d = '-'+d
        return(d)
# -4.296e-05 1.00045


def float_split(num: float):  # 暂时分割成两片
    if abs(num) > 0.01:
        return(round(num, 2))
    num_str = float_to_str(num)
    if num_str.find('e')-num_str.find('.') < 3 and abs(num) < 1:
        return(num_str)
    if 'e' in num_str:
        first_piece = num_str[:num_str.find(
            '.')+2]+num_str[num_str.find('e'):]
    else:
        first_piece = num_str[:-2]
        second_piece = num_str[4:6]
    return(first_piece)


def Encode(array):  # 将模型字典中的数字编码为字符串
    if not isinstance(array[0], list):
        for i, num in enumerate(array):
            array[i] = float_split(num)
    else:
        for subarray in array:
            Encode(subarray)


# 输入：待传输的模型参数（state_dict），原地更改：列表内容是str的模型参数


def Param_compression(dict: OrderedDict):
    for key, value in dict.items():
        temp = value.tolist()
        Encode(temp)
        dict[key] = temp


# ---------------------------------


def str_to_float(string: str):  # 将字符串转化为浮点数
    return(float(string))


def Decode(array: list):  # 将模型字典中的字符串编码为数字
    if not isinstance(array[0], list):
        for i, num in enumerate(array):
            array[i] = str_to_float(num)
    else:
        for subarray in array:
            Decode(subarray)


def Param_recovery(dict: OrderedDict):  # 此时值已经不是tensor了，而是列表
    for key, value in dict.items():
        temp = value
        Decode(value)  # 这里temp就是tensor
        dict[key] = torch.tensor(temp, dtype=torch.float32)
