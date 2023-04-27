import re

# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.model_selection import KFold
import pandas as pd

import random
# 引入AR模型的模块
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.api import qqplot
# 跨文件夹调用函数
# 使用sys
import sys
# sys.path.append(r'./model')                   #可以跨文件夹引用
# sys.path.append(r'./utils')
# from Dipca_demo import *
from model import Dipca_demo
from model import DiCCA_demo
from utils import dat
from utils import Func
from utils import filter
from utils import chabu
from model.cluster_zs import cluster_exp
from utils import corr
from utils import matrix
from scipy import interpolate
from sklearn.linear_model import LogisticRegression
import scipy.stats
import utils
# x=x_train[:,47:52]
# y=[1:1:100]

def visual(data,start,end,label):
    plt.figure(figsize=(8, 4), dpi=160)
    x = range(1,data.shape[0]+1)
    y1 = [int(i) for i in x]  # uniform提供[35, 40)随机值
    y2 = data[:,start:end]
    if y2.shape[1]!=len(label):
        label=label[:y2.shape[1]]
    plt.plot(y1, y2,label=label)
    plt.legend(loc='upper right')  # 右上角显示各个曲线代表什么意思
    plt.title('test05,冷凝器入水口')
    plt.show()

def dynamic_visual(T_test,V_test):
    plt.figure(figsize=(80, 60), dpi=200)
    if len(T_test.shape)==1:
        T_test=T_test.reshape([T_test.shape[0],1])
    # if len(V_test.shape) == 1:
    #     V_test = V_test.reshape([V_test.shape[0], 1])
    for i in range(T_test.shape[1]):
        plt.subplot(T_test.shape[1]+1,1,i+1)
        plt.plot(T_test[:,i],label=r"DLV{index}".format(index=i))
        # plt.plot(V_test[:, i], label=r"DLV{index}_deviation".format(index=i))
        plt.legend()
    plt.show()
    plot_pacf(np.squeeze(T_test[:, 0]), lags=20).show()
    plot_acf(np.squeeze(T_test[:, 0]), lags=20).show()
    # plot_pacf(np.squeeze(V_test[:,0]), lags=20).show()
    # plot_acf(np.squeeze(V_test[:,0]), lags=20).show()
    plt.show()
def model_result(model,X,order):
    predict = model.predict(start=order[0], end=X.shape[0]-1)
    # 计算残差 存在后移
    residual = X[order[0]:order[0]+predict[1:].shape[0]] - predict[1:]

    residual=X[order[0]:]-predict
    plt.figure(figsize=(8, 4), dpi=200)
    plt.subplot(2, 1, 1)
    plt.plot(predict[1:],label='pre')
    plt.plot(X[order[0]:],label='ori')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(residual)
    plt.show()
    return residual

def OI_calculate(X,MA_model,pdq_order):
    para_index = []
    for i in range(len(MA_model.param_names)):
        # 查找ma.L1这类参数，作为实际的平稳性标准输入
        result = re.search("ma.L", MA_model.param_names[i])
        if result != None:
            para_index.append(i)
    # 构建好index中依次从params中取出对应的值构成数组
    para = np.array([MA_model.params[i] for i in para_index])
    OI = np.sum(para ** 2) * (MA_model.params[para_index[0] + pdq_order[2]])
    OI= OI / np.var(X)
    print("白噪声的系数的平方和为{N_sum}，白噪声的方差为{N_var}".format(N_sum=np.sum(para ** 2),N_var=(MA_model.params[para_index[0] + pdq_order[2]])))
    print("实际曲线的方差为{Y_Var}".format(Y_Var=np.var(X)))
    print("OI={OI}\n".format(OI=OI))
    return OI


def data_pre(mode):
    if mode==1:
        x_train = dat.read_dat("./data/TE/train/d00.dat")
        x_test = dat.read_dat("./data/TE/test/d05_te.dat")
    elif mode == 2:
        x_train = loadmat(r"./data/DiPCA_Data.mat")["X"]
        x_test = loadmat(r"./data/DiPCA_Data_err1.mat")["X"]
        x_train=x_train[20:]
        x_test=x_test[20:]
    elif mode == 3:
        x_train = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmv.mat")["xmv"]
        x_test = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmeas.mat")["simout"]
        x_train=np.append(x_test,x_train,axis=1)
        x_test=x_train
    elif mode == 3.5:
        x_train = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmv_5_smooth.mat")["xmv"]
        x_test = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmeas_5_smooth.mat")["simout"]
        x_train = np.append(x_test, x_train, axis=1)
        x_test = x_train
    elif mode == 4:#G产物变多
        x_train = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmv_change.mat")["xmv"]
        x_test = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmeas_change.mat")["simout"]
        x_train = np.append(x_test, x_train, axis=1)
        x_test = x_train
    elif mode == 5:#G产物变少少少
        x_train = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmv_G_down.mat")["xmv"]
        x_test = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmeas_G_down.mat")["simout"]
        x_train = np.append(x_test, x_train, axis=1)
        x_test = x_train
    elif mode == 6:#G产物变少少少
        x_train = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmv_G_up_20h.mat")["xmv"]
        x_test = loadmat(r"C:\Users\zs\matlabProject\temexd_mod\data\xmeas_G_up_20h.mat")["simout"]
        x_train = np.append(x_test, x_train, axis=1)
        x_test = x_train
    # x_train=loadmat(r'./data/d00.mat')['d00']
    # x_test=loadmat(r'./data/d00te.mat')['d00te']
    return x_train,x_test

def calc_corr2(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = (sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b])) ** 0.5
    corr_factor = cov_ab / sq
    return corr_factor


if __name__ == "__main__":
    data1, data2 = data_pre(mode=3.5)
    data1=chabu.cubic3(data1,lag_same=25,loc_l=22,loc_r=40)
    data2=data1
    #挑选变量
    train_test_ratio=0.7
    xmv=data1[:,41:]
    xmv_indirect=data1[:,[0,1,2,3,4,9,13,16,18,20]]
    output=data1[:39]
    data=np.append(xmv,xmv_indirect)
