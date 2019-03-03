#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 19:34
# @Author  : Arrow and Bullet
# @FileName: semiconductor.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
import pca
from numpy import *


def replaceNanWithMeam():
    datMat = pca.loadDataSet("./data/secom.data", " ")
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


dataMat = replaceNanWithMeam()
# print(dataMat)

meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=False)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals)