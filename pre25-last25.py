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
from utils import chabu
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


def attention(X,T,l):
    # X与T的维数必须匹配
    Q = T
    # s = np.array([0.1,0.1,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 1])
    s = np.array([0.6,0.7, 0.8, 0.9, 1])
    for i in range(l, X.shape[0]):
        Q_t = Q[i - l:i, :]
        X_t = X[i - l:i, :]
        K_t = np.zeros([l, T.shape[1]])
        delta_t=np.zeros([l])
        for j in range(0, l):
            try:
                if Q_t.shape[0]<l-1:
                    raise Exception("sb_Q_t")
                K_t[j, :] = s[j] * Q_t[j, :]
                delta_t[j] = np.dot(Q_t[l-1, :], K_t[j, :])
            except Exception as e:
                print('第{i}个窗口，第{j}个时刻'.format(i=i, j=j))
                print('X与T的维数必须匹配')
        delta_t_softmax = Func.softmax(delta_t)
        # v_t = np.sum(X_t, axis=0)
        # v_t = v_t.reshape([1, X.shape[1]])
        # att = delta_t_softmax*v_t
        att=np.dot(delta_t_softmax,X_t)
        att=att.reshape([1,att.shape[0]])
        if i == l:
            X_att = att
        else:
            X_att = np.append(X_att, att, axis=0)
    return X_att

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
    # x_train=loadmat(r'./data/d00.mat')['d00']
    # x_test=loadmat(r'./data/d00te.mat')['d00te']
    # 熟悉一下字符串的拼接操作
    # visual(x_test,41, 52, ['D进料', 'E进料', 'A进料', '总进料','压缩机','排放流量','分离器流量','产品流量','汽提塔水流阀','反应器冷却水流量','冷凝器冷却水流量'])
    # visual(x_test,39, 41, ['G', 'H'])
    # visual(x_test,36, 39, ['D', 'E', 'F','223'])
    # X=loadmat(r"C:\Users\zs\matlabProject\biye\data_mat\DiPCA_Data.mat")['X']
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
    s_range = 5# 搜索0-4的范围
    a_range = 5
    fold = 5
    Mode=2
    # 1：TE数据；2参考数据
    #TE模型仿真数据 平稳的
    x_train, x_test = data_pre(mode=4)
    x_train=chabu.cubic3(x_train,lag_same=25,loc_l=22,loc_r=40)
    # xmeas=39
    x_test=x_train
    for xmeas in range(0,41):
        x_train=x_test
        # plt.plot(x_train[:,xmeas])
        # plt.show()
        x_train, x_test = Dipca_demo.normalize(x_train, x_test)
        # 选取想要的变量  检测变量在第一个
        # x_train=x_train[:, 41:]

        x_train=np.append(x_train[:,xmeas].reshape([x_train.shape[0],1]),x_train[:,41:] ,axis=1)
        lag=25#至少25个点，因为在成分检测中25次变一会
        # [s, a, press] = Dipca_demo.cv_DiPCA(x_train, s_range, a_range, fold)  # 交叉验证选取主元数
        s=3
        a=5
        DLV_index=1
        # print(s, a)
        P, W, T, V, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim,J_matrix= Dipca_demo.fit_DiPCA(x_train[:,:], s, a)  # 建模
        delta_dynamic = []
        for i in range(0,x_train.shape[0],lag):
            temp=x_train[:i+lag,:]
            # P, W, T, V, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim = Dipca_demo.fit_DiPCA(temp, s, a)  # 建模
            phi_v_index, phi_s_index, T_test, V_test, Xe_test = Dipca_demo.test_DiPCA(temp, P, W, Theta_hat, s, PHI_s,
                                                                                      PHI_v)  # 测试
            if i==0:delta_dynamic=T_test[:,DLV_index-1]
            else:
                delta_dynamic=np.append(delta_dynamic,T_test[i:i+lag,DLV_index-1],axis=0)
                delta_dynamic[i-lag:i]=delta_dynamic[i:i+lag]-delta_dynamic[i-lag:i]
                # delta_dynamic[i - lag:i]=delta_dynamic[i:i+lag]/delta_dynamic[i-lag:i]
        # print(phi_v_lim, phi_s_lim)
        Xe_test_T, T_weight = Dipca_demo.PCA(Xe_test)

        output=x_train[: ,0]
        delta_output=[]
        for i in range(0,x_train.shape[0]-lag,lag):
            temp = output[i + lag:i + 2*lag]
            delta_output[i:i + lag]=temp-output[i:i + lag]

        # delta_output=output[1:500]-output[0:499]
        # for i in range(0,output.shape[0]):
        #     output[i]=output[i+1]-output[i]

        corr_matri = np.array([1])
        for i in range(0,x_train.shape[0]-lag,lag):
            corr = calc_corr2(delta_dynamic[i:i + lag], output[i:i + lag])
            corr_matri=np.append(corr_matri,np.array([corr]),axis=0)
            # print("相关系数{corr}".format(corr=calc_corr2(delta_dynamic[i:i+lag], output[i:i+lag])))
        print("xmeas{num}的相关系数{corr}".format(corr=calc_corr2(delta_dynamic[100:2500], delta_output[100:2500]),num=xmeas+1))

        plt.figure(figsize=(12, 8), dpi=400)
        # plt.title("delta{i}".format(i=calc_corr2(delta_dynamic[0:1000], delta_output[0:1000])),fontdict={'family': 'serif', 'size': 18, 'color': 'red', 'weight': 'bold'})
        plt.subplot(3, 1, 1)
        plt.plot(T_test[100:2500,DLV_index-1])
        plt.plot(output[100:2500])
        plt.ylabel("delta\nDLV{DLV_index}\n{i}".format(DLV_index=DLV_index,i=calc_corr2(T_test[100:2500,DLV_index-1], output[100:2500])),rotation=0)
        fig = plt.subplot(3, 1, 2)
        plt.plot(delta_dynamic[100:2500], label="xmeas"+str(xmeas+1))
        plt.plot(delta_output[100:2500], label='O')
        plt.ylabel("delta\nDLV{DLV_index}\n{i}".format(DLV_index=DLV_index,i=calc_corr2(delta_dynamic[100:2500], delta_output[100:2500])),rotation=0)
        plt.legend(loc='upper right')
        plt.subplot(3, 1, 3)
        plt.plot(Xe_test_T[100:2500,0])
        plt.savefig("./figure/"+"G阶跃1/DLV{DLV_index}/DLV{DLV_index}_".format(DLV_index=DLV_index)+str(xmeas+1)+".jpg")  ## 保存图片
        plt.close()
        # plt.show()

    # 是否采用回归分析
    # modelLR = LogisticRegression()
    # modelLR.fit(delta_dynamic, output)

    # Dipca_demo.visualization_DiPCA(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim)  # 监测结果可视化