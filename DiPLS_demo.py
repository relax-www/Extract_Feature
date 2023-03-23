from DiPLS_FaultDection import DiPLS_FaultDection
from scipy.io import loadmat
import numpy as np
import dat

path_train = "./TE/train/d00.dat"
path_test= "./TE/test/d12_te.dat"

data1=dat.read_dat(path_train)
data2=dat.read_dat(path_test)
# data1 = loadmat(path_train)['d00te']

X1 = data1[:,:22]
X2 = data1[:,-11:]
X_Train= np.hstack((X1,X2))
Y_Train = data1[:,34:36]

# data2 = loadmat(path_test)['d05te']

X11 = data2[:,:22]
X22 = data2[:,-11:]
X_test = np.hstack((X11,X22))
Y_test  = data2[:,34:36]

#初始化
DiPLS_FaultDection = DiPLS_FaultDection(s=4,a=2)

#数据标准化（若是标准化过后的数据则无需这一步）
[X_Train,X_test] = DiPLS_FaultDection.normalize(X_Train,X_test)
[Y_Train,Y_test] = DiPLS_FaultDection.normalize(Y_Train,Y_test)


#训练模型
model = DiPLS_FaultDection.train(X_Train,Y_Train)
"""
            self.model = {
            'R': R,
            'T_y': T_y,
            'Q_y': Q_y,
            'P_y': P_y,
            'T_o': T_o,
            'P_o': P_o,
            'T_r': T_r,       
            'P_r': P_r,
            'T_y_lim': T_y_lim,
            'T_o_lim': T_o_lim,
            'T_r_lim': T_r_lim,
            'Qr_lim': Qr_lim             
            }  
"""

#测试模型
testresult = DiPLS_FaultDection.test(model,X_test)

"""
         testresult = {
        'T_y_values': T_y_values,
        'T_o_values': T_o_values,
        'T_r_values': T_r_values,
        'Q_r_values': Q_r_values
        }
"""
#检测结果可视化
DiPLS_FaultDection.visualization(model,testresult)
DiPLS_FaultDection.predict(model,X_test)