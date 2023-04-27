
import os
import numpy as np
import tensorflow
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
# import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # 交叉验证法的库
import openpyxl as xl
import time

from scipy.io import loadmat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def caculate_RMSE(y_true, y_pred):
    """
    This is a groups style docs.

    Parameters:
      param1 - y的真实值
      param2 - 预测值
    Returns:
    	RMSE一个数值
    """
    return pow(abs(keras.losses.mean_squared_error(y_true, y_pred)), 0.5)

def Model_lstm(n_feature):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=hidden_node, activation='tanh', recurrent_activation="sigmoid"
                                ,kernel_regularizer=keras.regularizers.l1(0.01),input_shape=(time_step, n_feature)))
    # model.add(keras.layers.LSTM())input_dim=30,input_length=4,
    # ten units LSTM
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0
                                 , amsgrad=False)
    # 文中使用RMSE和MAPE
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['Accuracy', 'mae'])

    return model


def train_model(model,x_train,y_train,train_history_path):
    avg_accuracy = 0
    avg_loss = 0
    # 过拟合常用正则和早停，文中使用早停=30，其余参数未知
    # verbose = 0 不在标准输出流输出日志信息
    # verbose = 1 输出进度条记录，进度条如 [====>…] - ETA
    # verbose = 2 每个epoch输出一行记录
    Early_Stop = keras.callbacks.EarlyStopping(monitor='Accuracy', patience=20, verbose=2, mode='auto')

    # train_x=[],train_y=[],test_x=[],test_y=[]
    # kfold = KFold(n_splits=5, shuffle=False)  # 由于是LSTM的代码所以不需要random_state
    # i=0
    # print(time_step,learn_rate,hidden_node,sep=' ')
    # for train_index, test_index in kfold.split(x_train, y_train):
    #     i+=1
    #     print("Time_Step")
    #     print("5折交叉验证法   第%d"%i+"次")
    #     train_x, test_x = x_train[train_index], x_train[test_index]
    #     train_y, test_y = y_train[train_index], y_train[test_index]
    #
    #     # 此处  batch_size=16  为自己设置，没有依据，瞎蒙了一个   最后batch_size=1太慢了，还是16
    #     hist = model.fit(train_x, train_y, batch_size=128, epochs=60,verbose=0)#,callbacks=[Early_Stop])
    #     ###################################            5折验证法          ##########################3
    #     print('Model evaluation: ', model.evaluate(test_x, test_y))
    #     avg_accuracy += model.evaluate(test_x, test_y)[1]
    #     avg_loss += model.evaluate(test_x, test_y)[0]
    model.fit(x_train, y_train, batch_size=128, epochs=60, verbose=1)  # ,callbacks=[Early_Stop])
    print("K fold average accuracy: {}".format(avg_accuracy / n_split))
    print("K fold average loss: {}".format(avg_loss / n_split))
    # frame_train_acc = pd.DataFrame({'1':[time_step],'2':[learn_rate],'3':[hidden_node]
    #                                    ,'acc':[avg_accuracy / n_split], 'loss':[avg_loss / n_split]})
    # with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
    #     if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
    #         excel_data = pd.read_excel(train_history_path, sheet_name="train")
    #         Start_Col = excel_data.shape[1]
    #         Start_Row = excel_data.shape[0]
    #     else:
    #         Start_Col = 0
    #         Start_Row = 0
    #     frame_train_acc.to_excel(writer, sheet_name="train", encoding='utf-8', startcol=0, startrow=Start_Row+1,
    #                              index=False,header=False)
    return model


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]-1])
        dataY.append(dataset[i, -1])
    return np.array(dataX), np.array(dataY)





Time_Step = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Learn_Rate = [0.1, 0.01, 0.001, 0.0001]
Hidden_Node = [16, 32, 64, 128, 256]

n_split = 5
time_step = 10
learn_rate = 0.01
hidden_node = 256
train_history_path="./train_history.xlsx"
resource_path="./train_history.xlsx"

if __name__=="__main__":

    # for sttt in range(100, 7100, 200):
    #     b = np.loadtxt("./data/R_Output{i}.txt".format(i=sttt), delimiter=',')
    #     if sttt==100:
    #         data=b
    #     else:data=np.append(data,b,axis=0)
    # np.savetxt("./data/final.txt", data, fmt='%f', delimiter=',')
    data = np.loadtxt("./data/final.txt", delimiter=',')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    train=data[:2500,:]
    test=data[2500:,:]
    X_train=train[:,:-1]
    y_train=train[:,-1]
    X_test=test[:,:-1]
    y_test=test[:,-1]
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score
    alpha = 0.1
    lasso = Lasso(alpha=alpha)

    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    plt.plot(y_pred_lasso)
    plt.plot(y_test)
    plt.show()
    print(lasso)
    print("r^2 on test data : %f" % r2_score_lasso)