import re

# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

def PCA(X):
    a_v = Dipca_demo.pc_number(X)
    # Sv是SVD的特征值
    _, Sv, Pv = np.linalg.svd(X)  # Sv是一维的
    Pv = Pv.T
    Pv = Pv[:, 0:a_v]
    Ts=np.dot(X,Pv)
    PCA_eigenvalue=Sv**2
    eigenvalue_wight=PCA_eigenvalue/np.sum(PCA_eigenvalue)
    # 此处进需要得分向量则其他部分不予输出
    return Ts,eigenvalue_wight[0:a_v]

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

if __name__ == "__main__":
    s_range = 3# 搜索0-4的范围
    a_range = 3
    fold = 5
    Mode=2
    # 1：TE数据；2参考数据

    x_train, x_test = data_pre(3.5)
    x_train=x_train[:1000,:]
    x_test = x_test[1000:1200, :]
    x_train, x_test = Dipca_demo.normalize(x_train, x_test)

    [s, a, press] = Dipca_demo.cv_DiPCA(x_train, s_range, a_range, fold)  # 交叉验证选取主元数
    P, W, T, V, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim,J_matrix = Dipca_demo.fit_DiPCA(x_train, s, a)  # 建模
    phi_v_index, phi_s_index, T_test, V_test, Xe_test = Dipca_demo.test_DiPCA(x_train, P, W, Theta_hat, s, PHI_s,
                                                                              PHI_v)  # 测试
    print(phi_v_lim, phi_s_lim)
    print(s, a)
    # Dipca_demo.visualization_DiPCA(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim)  # 监测结果可视化
    # 三位柱状图
    scatter_data=np.zeros([press.shape[0]*press.shape[1],3])
    for i in range(0,press.shape[0]):
        for j in range(0, press.shape[1]):
            scatter_data[i*press.shape[0]+j,0]=i+1  #s阶数
            scatter_data[i * press.shape[0] + j, 1] =j+1#a潜变量个数
            scatter_data[i * press.shape[0] + j, 2] =press[i,j]#准确率

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(scatter_data[:,0], scatter_data[:,1], scatter_data[:,2],s=100)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('a', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('s', fontdict={'size': 15, 'color': 'red'})
    fig.add_axes(ax)
    plt.show()


    # ##############################################################################################################
    pingwenxing=1
    if pingwenxing==1:
        # 动态部分平稳性
        # dynamic_visual(V_test, V_test)
        # 关于此处参数设置的问题还需要仔细斟酌V
        # V_test= Dipca_demo.normalize(V_test)
        # V_test=V_test[1:]-V_test[0:V_test.shape[0]-1]

        # 动态部分平稳性
        pdq_order=[0,0,20]
        # ARIMA会自动进行参数搜索，范围是从零到输入的整数
        OI_dynamic=np.zeros([a])
        for i in range(0,a):
            ARMA_model = ARIMA(T_test[:,i], order=pdq_order).fit()
            residual=model_result(ARMA_model,T_test[:,i], pdq_order) #这是用于预测的检验一般不使用
            OI_dynamic[i] = OI_calculate(T_test[:,i], ARMA_model, pdq_order)*J_matrix[i]/np.sum(J_matrix)
        OI_dynamic = np.sum(OI_dynamic, axis=0)
        print("动态部分稳定性：{o}".format(o=OI_dynamic))
        # plt.plot(residual-V_test[:,0])
        # plt.show()
        # error = residual[np.abs(residual - np.mean(residual)) > 3 * np.std(residual)]
        # residual = residual[np.abs(residual - np.mean(residual)) <= 3 * np.std(residual)]
        # plot_acf(residual).show()
        # plot_pacf(residual).show()

        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # fig = qqplot(V_test[:,0], line='q', ax=ax, fit=True)
        # plt.show()
        # plt.hist(V_test[:,0], bins=20, edgecolor='white')
        # plt.show()
        # MA_model= ARIMA(residual[:], order=[0,0,10]).fit()
        # OI_dynamic = OI_calculate(residual[:], MA_model, [0,0,10])
        # residual = model_result(MA_model, residual[:], [0,0,10])
        print(np.var(V_test[:, 0]))

        # 静态部分平稳性
        Xe_test_T,T_weight=PCA(Xe_test)
        print(Xe_test_T.shape[1])
        # dynamic_visual(np.array(Xe_test_T)[:,0], np.array(Xe_test_T)[:,0])
        OI_static=0
        for i in range(T_weight.shape[0]):
            pdq_order_static = [0, 0, 30]
            MA_model = ARIMA(Xe_test_T[:,i], order=pdq_order_static).fit()
            temp = OI_calculate(Xe_test_T[:, i], MA_model, pdq_order_static)
            OI_static=OI_static+T_weight[i]*temp
            # model_result(MA_model, Xe_test_T[:, 0], pdq_order_static)  # 这是用于预测的检验一般不使用
        OI_static=OI_static/np.sum(T_weight)

        print("动态部分稳定性{dynamic}，静态部分稳定性{static}".format(dynamic=OI_dynamic,static=OI_static))
        print("整体的稳定性{whole}".format(whole=OI_dynamic + OI_static))
    #################################################################################################################
    anquan=1
    if anquan == 1:
        #3关于安全性的估计
        # 定义实际上的误差
        loc = np.where(phi_v_index == np.min(phi_v_index))
        a = np.ones(len(phi_v_index))
        b = np.ones(len(phi_s_index))
        SI_v=np.zeros(phi_v_index.shape[0])
        SI_s=np.zeros(phi_s_index.shape[0])
        for i in range(1, phi_v_index.shape[0]):
            a=phi_v_lim - np.min(phi_v_index)
            if a !=0:
                SI_v[i]= (phi_v_lim - phi_v_index[i]) / (phi_v_lim - np.min(phi_v_index))
                # SI_v[i] = (phi_v_lim - phi_v_index[i]) / phi_v_lim#就当做是安全性能指标来计算了，我不想别的了
            else: SI_v[i]=1
        for i in range(1, phi_s_index.shape[0]):
            b=phi_s_lim - np.min(phi_s_index)
            if b !=0 :
                SI_s[i]= (phi_s_lim - phi_s_index[i]) / (phi_v_lim - np.min(phi_s_index))
                # SI_s[i] = (phi_s_lim - phi_s_index[i]) / phi_s_lim#就当做是安全性能指标来计算了，我不想别的了
            else:SI_s[i]=1

        error = SI_v[np.abs(SI_v - np.mean(SI_v)) > 3 * np.std(SI_v)]
        SI_v = SI_v[np.abs(SI_v - np.mean(SI_v)) <= 3 * np.std(SI_v)]
        error = SI_s[np.abs(SI_s - np.mean(SI_s)) > 3 * np.std(SI_s)]
        SI_s = SI_s[np.abs(SI_s - np.mean(SI_s)) <= 3 * np.std(SI_s)]
        plt.figure(figsize=(8,4),dpi=200)
        ax1=plt.subplot(2,1,1)
        plt.plot(SI_v)
        plt.plot(np.zeros(phi_v_index.shape[0]))
        ax1.set_title('dinamic')
        ax2=plt.subplot(2,1,2)
        plt.plot(SI_s)
        plt.plot(np.zeros(phi_s_index.shape[0]))
        ax2.set_title('static')
        plt.show()
    #################################################################################################################

