#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 17:31
# @Author  : Arrow and Bullet
# @FileName: linalg.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
from numpy import *

A = mat("1 -2 1;0 2 -8")
print(A)  # [[ 1 -2  1][ 0  2 -8]]
B = linalg.pinv(A)
print(B)  # [[ 0.25757576  0.04545455][-0.42424242 -0.04545455][-0.10606061 -0.13636364]]

