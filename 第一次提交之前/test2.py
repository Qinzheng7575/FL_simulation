import torch
import struct
import numpy as np


def fun(name, age, **kw):
    a = name
    b = age
    # print(kw)
    if kw['method'] == 'base':
        print(kw['method'])


fun('qin', 18, method='base')
