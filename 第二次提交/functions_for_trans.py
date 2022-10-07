from collections import OrderedDict
import torch


def float_to_str(num: float):  # Convert float scientific notation to string
    d = str(num)
    neg = True if d[0] == '-' else False  # determine if it is negative
    point_pos = d.find('.')

    # No need for scientific notation
    neednt = False if abs(num) > 1 or 'e' in d else True
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


def float_split(num: float):  # Split out the base value
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


def Encode(array):  # Encode numbers in model as strings
    if not isinstance(array[0], list):
        for i, num in enumerate(array):
            array[i] = float_split(num)
    else:
        for subarray in array:
            Encode(subarray)


def Param_compression(dict: OrderedDict):  # Inplace change model parameters
    for key, value in dict.items():
        temp = value.tolist()
        Encode(temp)
        dict[key] = temp
# -------------------------------------------------


def str_to_float(string: str):  # Convert string to float
    return(float(string))


def Decode(array: list):  # Encode strings in model as numbers
    if not isinstance(array[0], list):
        for i, num in enumerate(array):
            array[i] = str_to_float(num)
    else:
        for subarray in array:
            Decode(subarray)


def Param_recovery(dict: OrderedDict):  # Model parameter recovery
    for key, value in dict.items():
        temp = value
        Decode(value)
        dict[key] = torch.tensor(temp, dtype=torch.float32)
