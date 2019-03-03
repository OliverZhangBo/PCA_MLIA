#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 19:18
# @Author  : Arrow and Bullet
# @FileName: run.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
import pca
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

dataMat = pca.loadDataSet("./data/testSet.txt")


lowDMat, reconMat = pca.pca(dataMat, 2)
m, n = shape(lowDMat)

print(m, n)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker="^", s=90)
ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50, c="red")
fig.show()