
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
from sklearn.model_selection import KFold
import pandas as pd
import dat
import random

# 跨文件夹调用函数
# 使用sys
import sys
sys.path.append(r'./SAE')                   #可以跨文件夹引用
# from Dipca_demo import *
import Dipca_demo



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


if __name__ == "__main__":

    a=np.array([1,2,3,4])
    b=a**2
    x_train = dat.read_dat("./TE/train/d05.dat")
    x_test = dat.read_dat("./TE/test/d05_te.dat")
    # visual(x_test,41, 52, ['D进料', 'E进料', 'A进料', '总进料','压缩机','排放流量','分离器流量','产品流量','汽提塔水流阀','反应器冷却水流量','冷凝器冷却水流量'])
    # visual(x_test,39, 41, ['G', 'H'])
    # visual(x_test,36, 39, ['D', 'E', 'F','223'])
    X=loadmat(r"C:\Users\zs\matlabProject\biye\data_mat\DiPCA_Data.mat")['X']
    s_range = 1# 搜索0-4的范围
    a_range = 3
    fold = 5
    [s, a] = Dipca_demo.cv_DiPCA(X[:2000,:], s_range, a_range, fold)  # 交叉验证选取主元数
    # P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim = DiPCA(X_train, s, a);#建模
    P, W, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim = Dipca_demo.fit_DiPCA(X[:2000,:], s, a);  # 建模
    phi_v_index, phi_s_index = Dipca_demo.test_DiPCA(X[2000:,:], P, W, Theta_hat, s, PHI_s, PHI_v);  # 测试
    print(phi_v_lim, phi_s_lim)
    print(s, a)
    # DiPCA_visualization(phi_v_index,Ts_index,Qs_index,phi_v_lim,Ts2_lim,Qs_lim);# 监测结果可视化
    Dipca_demo.visualization_DiPCA(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim);  # 监测结果可视化
    # 定义实际上的误差
    loc = np.where(phi_v_index == np.min(phi_v_index))
    a = np.ones(len(phi_v_index))
    b = np.min(phi_v_index)
    SI_v=np.zeros(phi_v_index.shape[0])
    for i in range(1, phi_v_index.shape[0]):
        a=phi_v_index[i] - np.min(phi_v_index[:i])
        if a!= 0 :
            SI_v[i]= (phi_v_lim - phi_v_index[i]) / (phi_v_index[i] - np.min(phi_v_index[:i]))
            SI_v[i] = (phi_v_lim - phi_v_index[i]) / phi_v_lim#就当做是安全性能指标来计算了，我不想别的了
        # else
    plt.figure(figsize=(8,4),dpi=200)
    plt.plot(SI_v)
    plt.show()
    # for i in range(2000,X.shape[0]):
    #     R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    #     T=np.dot(X(i),R)
    #
    #     n = X.shape[0]
    #     N = n - s
    #     R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    #     if s > 0:
    #         T = np.dot(X, R)
    #         TTs = T[s:N + s, :]
    #         TT = T[0:N, :]
    #         i = 1
    #         while i < s:
    #             Ts = T[i:N + i, :]
    #             TT = np.c_[TT, Ts]
    #             i = i + 1
    #         TTshat = np.dot(TT, Theta)
    #     phi_v_index = np.zeros(N)
    #     phi_s_index = np.zeros(N)
    #     k = s
    #     while k < s + N:
    #         if s > 0:
    #             temp = TTs[k - s, :] - TTshat[k - s, :]
    #             temp = np.array([temp])
    #             v = temp.T
    #             phi_v_index[k - s] = np.dot(np.dot(v.T, PHI_v), v)
    #             e = X[k - s, :].T - np.dot(P, TTshat[k - s, :].T)
    #         else:
    #             e = X[k - s, :].T
    #         # Ts_index[k-s] = np.dot(np.dot(e.T, Mst), e)
    #         # Qs_index[k-s] = np.dot(np.dot(e.T, Msq), e)
    #         phi_s_index[k - s] = np.dot(np.dot(e.T, PHI_s), e)
    #         k = k + 1
    #     # return phi_v_index,Ts_index,Qs_index