import re

# _*_ coding:utf-8 _*_

import numpy as np
import scipy.io as sio
from scipy import stats

ek=stats.norm.rvs(0,0.01,size=[3000,5])
vk=stats.norm.rvs(0,0.01,size=[3000,3])

A=[[0.5205,0.1022,0.0599],
    [0.5367,-0.0139,0.4159],
    [0.0412,0.6054,0.3874]]

P=[ [0.4316,0.1723,-0.0574],
    [0.1202,-0.1463,0.5348],
    [0.2483,0.1982,0.4797],
    [0.1151,0.1557,0.3739],
    [0.2258,0.5461,-0.0424]]

c=[0.5205 ,0.5367 ,0.0412]

t=[[1],[1],[1]]

A=np.array(A)
P=np.array(P)
c=np.array(c).T
t=np.array(t)

def data_generation(A,P,c,t,ek,vk):
    for i in range(0,3000):
        if i>0:
            temp_t=c+A@t[:,i-1]+vk[i,:].T
            temp_t=np.reshape(temp_t,[3,1])
            t=np.append(t,temp_t,axis=1)
        if i==0:
            xk=P@t[:,i]+ek[i,:].T
            xk = np.reshape(xk, [5, 1])
        else:
            xk_temp=P@t[:,i]+ek[i,:].T
            xk_temp = np.reshape(xk_temp, [5, 1])
            xk=np.append(xk,xk_temp,axis=1)
    xk=np.array(xk)
    X=xk.T
    X={'X':X}
    sio.savemat('./DiPCA_Data.mat', X)
    return X
def data_generation_err1(A,P,c,t,ek,vk):
    error=np.array([3,0,0,0,0])
    error = np.reshape(error, [5, 1])
    for i in range(0,1000):
        if i>0:
            temp_t=c+A@t[:,i-1]+vk[i,:].T
            temp_t=np.reshape(temp_t,[3,1])
            # if i > 200:temp_t+=error
            t=np.append(t,temp_t,axis=1)
        if i==0:
            xk=P@t[:,i]+ek[i,:].T
            xk = np.reshape(xk, [5, 1])
        else:
            xk_temp = P@t[:, i] + ek[i, :].T
            xk_temp = np.reshape(xk_temp, [5, 1])
            if i > 200: xk_temp += error
            xk=np.append(xk,xk_temp,axis=1)
    xk=np.array(xk)
    X=xk.T
    X={'X':X}
    sio.savemat('./DiPCA_Data_err1.mat', X)
    return X
if __name__=="__main__":
    data_generation(A,P,c,t,ek,vk)
    data_generation_err1(A,P,c,t,ek,vk)


