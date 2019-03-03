#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 16:41
# @Author  : Arrow and Bullet
# @FileName: pca.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
from numpy import *


def loadDataSet(fileName, delim="\t"):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]  # 直接利用列表生成式生成了二级列表，不知道我这样的称呼是否正确
    datArr = [list(map(float, line)) for line in stringArr]
    # 首先仍然是列表生成式遍历了每一行，然后用map函数把每一行进行了遍历，666666
    return mat(datArr)  # 矩阵转化


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)  # 按列求均值
    meanRemoved = dataMat - meanVals  # 数据减去均值
    covMat = cov(meanRemoved, rowvar=0)  # 求协方差,列为变量X，也就是每一个特征是变量
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算covMat的特征值和特征向量
    eigValInd = argsort(eigVals)  # 特征值排序 的索引值 默认从小到大的方式
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 按之前的索引值逆序排列特征值 得到的依旧是索引值呀
    # eigValInd = sorted(eigvals, reverse=True)
    redEigVects = eigVects[:, eigValInd]  # 特征向量排序 降序了现在是 因为现在的eigValInd就是降序的
    lowDDataMat = meanRemoved * redEigVects  # 移除均值后的原始数据 * 特征向量  这是矩阵利用N个特征（向量）将原始数据转化到新空间中
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 这是原始数据重构后
    # 这里还是不太明白的，用移除均值后的原始数据 * 特征向量 * 特征向量.T 就时原始数据的重构了 不明白
    return lowDDataMat, reconMat  #


