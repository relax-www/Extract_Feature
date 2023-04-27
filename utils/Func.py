import numpy as np
def softmax(x):
    # 计算每行的最大值
    row_max = np.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    data=np.random.randint(0,27,(9,3))
    x=data[:,0]
    y=data[:,1]
    z=data[:,2]

    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(x,y,z,c='r')
    fig.add_axes(ax)
    plt.show()
    # -*- coding: utf-8 -*-
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    length = 100
    x = []
    x = [i for i in range(-1 * length, length)]
    Y = X = np.array(x)
    X, Y = np.meshgrid(X, Y)
    z = X * X + Y * Y
    # z = 5*X*X + 8*Y*Y*Y
    Z = np.array(z).reshape(length * 2, length * 2)

    # 绘制表面
    fig = plt.figure()

    ax = fig.add_subplot(projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # 添加将值映射到颜色的颜色栏
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.add_axes(ax)
    plt.show()
    # ##############################################################################################################
    pingwenxing=0