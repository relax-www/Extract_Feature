# 综合分类数据集
import sys
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import cluster
#聚合聚类
from sklearn.cluster import AgglomerativeClustering
# 亲和力聚类
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
sys.path.append(".../utils")
from  utils import corr
from model import Dipca_demo
# 定义数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 为每个类的样本创建散点图
def dataset(X,y):
    for class_value in range(2):
        # 获取此类的示例的行索引
        row_ix = np.where(y == class_value)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        # 绘制散点图
    plt.show()

def Dbscan(X):#函数里不可以调用某些写法 如cluster.DBSCAN
    model =DBSCAN(eps=0.30, min_samples=9)
    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = np.unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    plt.show()


def K_means(X):
    model = KMeans(n_clusters=2)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = np.unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    plt.show()

def Guass(X):
    model = GaussianMixture(n_components=2)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = np.unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    plt.show()

def read_write_txt(flag,path,X):
    # 读取
    if flag==0:
        np.savetxt(path, X, fmt='%f', delimiter=',')
    else:
        b = np.loadtxt(path, delimiter=',')
    return b
#
#
#
def GaussianMixture_extract(X,jiangge,cluster_num,pic_flag,i=0):
    """
    This is a groups style docs.

    Parameters:
      param1 - 输入矩阵 200*25 200是样本，25是间隔
      param2 - 200个样本不可能全用，采样的频率
      param3 - 聚类分成几类
      param4 - 是否展示图片 0 不展示， 1展示
    Returns:
    	一条1*25的曲线

    Raises:
    	KeyError - raises an exception
    """
    X = X[range(0,X.shape[0], jiangge), :]
    X = (X - X.mean()) / X.std()

    # cluster_num = 3
    model = GaussianMixture(n_components=cluster_num)
    # model = DBSCAN(eps=0.30, min_samples=6)
    # # 模型拟合
    # model.fit(X)
    # # 为每个示例分配一个集群
    # yhat = model.predict(X)
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = np.unique(yhat)
    cluster_num = clusters.size
    # 为每个群集的样本创建散点图
    ratio = np.zeros(clusters.size)
    # plt.figure(figsize=(8, 6), dpi=150)
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)[0]#返回的是一个元组，而非数组
        ratio[cluster]=row_ix.size
    cluster1_idx = np.where(ratio == sorted(np.unique(ratio))[-1])[0][0]
    cluster2_idx = np.where(ratio == sorted(np.unique(ratio))[-2])[0][0]
    if cluster2_idx.size > 1:
        cluster2_idx = cluster2_idx[0]
    row_ix = np.where(yhat == cluster1_idx)[0]  # 返回的是一个元组，而非数组
    cluster1 = X[row_ix, :]
    row_ix = np.where(yhat == cluster2_idx)[0] # 返回的是一个元组，而非数组
    cluster2 = X[row_ix, :]
    cluster1_mean = np.mean(cluster1, axis=0)
    cluster2_mean = np.mean(cluster2, axis=0)
    relation = corr.calc_corr2(cluster1_mean, cluster2_mean)
    if relation > 0.9:
        flag = 1
    elif relation < -0.9:
        flag = -1
    else:
        flag = 0

    result=(cluster1_mean + (flag) * cluster2_mean) / (2 - int(flag == 0))

    if pic_flag==1:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(cluster1_mean, label="cluster1")
        plt.plot(cluster2_mean, label="cluster2")
        plt.plot(result, label="mean")
        plt.legend(loc="center right")
        plt.ylabel("relation\n{:.3f}".format(relation), rotation=0)
        plt.savefig("./figure/" + "{i}.jpg".format(i=i+1))
        plt.figure()
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = np.where(yhat == cluster)[0]  # 返回的是一个元组，而非数组。需要有【0】取出第一个数
            ratio[cluster] = row_ix.size
            plt.subplot(cluster_num, 1, cluster + 1)
            a = np.squeeze(X[row_ix, :])
            plt.plot(a.T)
            plt.grid()
            plt.savefig("./figure/" + "{i}.jpg".format(i=i))
        # plt.show()
    print(relation)
    return result


if __name__=="__main__":
    # # 定义数据集
    # X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
    #                            random_state=4)
    #
    #
    # data=read_write_txt(1,"./data/delta_dynamic700.txt",X)
    # GaussianMixture_extract(data, jiangge=10, cluster_num=3, pic_flag=1)
    for i in range(100,7100,200):
        data = read_write_txt(1, "./data/delta_dynamic{i}.txt".format(i=i), X)
        # GaussianMixture_extract(data, jiangge=10, cluster_num=3, pic_flag=1,i=i)


    # s=1
    # a=1
    # X_train, X_test = Dipca_demo.normalize(data.T, data.T)
    # s_range = 3
    # a_range = 5
    # fold = 5
    # [s, a, press] = Dipca_demo.cv_DiPCA(X_train, s_range, a_range, fold)  # 交叉验证选取主元数
    # # P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim = DiPCA(X_train, s, a);#建模
    # P, W, T, V, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim = Dipca_demo.fit_DiPCA(X_train, s, a)  # 建模
    # phi_v_index, phi_s_index, T_test, V_test, Xe_test = Dipca_demo.test_DiPCA(X_test, P, W, Theta_hat, s, PHI_s, PHI_v)  # 测试
    # # plt.plot(T_test.sum(axis=1))
    # plt.plot(T_test[:,0])
    # plt.show()
    # print(phi_v_lim, phi_s_lim)
    # print(s, a)




