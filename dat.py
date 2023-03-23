import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
from sklearn.model_selection import KFold
import pandas as pd
import pickle
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def read_all_data(path_test,path_train):
    '''
    读取TE过程的所有.dat数据并存人DataFrame中，输入参数为测试数据和训练数据的绝对路径
    '''
    var_name = []
    for i in range(1,42):
        var_name.append('XMEAS(' + str(i) + ')')
    for i in range(1,12):
        var_name.append('XMV(' + str(i) + ')')
    data_test, data_train = [], []
    # path_test = r'C:\Users\17253\Desktop\组内\K_shape\data\TE\test'
    # path_train = r'C:\Users\17253\Desktop\组内\K_shape\data\TE\train'
    test_join = glob.glob(os.path.join(path_test,'*.dat'))
    train_join = glob.glob(os.path.join(path_train,'*.dat'))
    for filename in test_join:
        data_test.append(pd.read_table(filename, sep = '\s+', header=None, engine='python', names = var_name))
    for filename2 in train_join:
        data_train.append(pd.read_table(filename2, sep = '\s+', header=None, engine='python', names = var_name))
    return data_test, data_train
def read_dat(PATH):
    f=open(PATH,encoding='utf-8')
    sentimentlist = []
    for line in f:
        s=line.strip()
        s = s.split("  ")
        s = [float(z) for z in s]
        sentimentlist.append(s)
    f.close()
    read_data=np.array(sentimentlist)#480*52
    if read_data.shape[0]<read_data.shape[1]:
        read_data=read_data.T
    return read_data


# x_test=dat.read_dat("./TE/test/d12_te.dat")
# data=read_dat("./TE/test/d12_te.dat")
# plt.figure(figsize=(3, 3), dpi=300)
#
# ax1 = plt.subplot(3, 1, 1)
# ax1.plot(data[:,0])
# ax1.set_xlabel('Samples')
# ax1.set_ylabel('$\phi_v$')
# ax1.set_title('monitor')
#
# ax2 = plt.subplot(3, 1, 2)
# ax2.plot(data[:,1])
# ax2.set_xlabel('Samples')
# ax2.set_ylabel('$\phi_s$')
#
# ax3 = plt.subplot(3, 1, 3)
# ax3.plot(data[:,2])
# ax3.set_xlabel('Samples')
# ax3.set_ylabel('$\phi_v$')
#
#
# plt.show()

















