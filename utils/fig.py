import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

press=np.random.randint(0,27,(3,3))
scatter_data = np.zeros([press.shape[0] * press.shape[1], 3])
for i in range(0, press.shape[0]):
    for j in range(0, press.shape[1]):
        scatter_data[i * press.shape[0] + j, 0] = i + 1  # s阶数
        scatter_data[i * press.shape[0] + j, 1] = j + 1  # a潜变量个数
        scatter_data[i * press.shape[0] + j, 2] = press[i, j]  # 准确率

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scatter_data[:, 0], scatter_data[:, 1], scatter_data[:, 2],s=100)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('a', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('s', fontdict={'size': 15, 'color': 'red'})
fig.add_axes(ax)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(range(1, 4), range(1, 4))
surf = ax.plot_surface(X, Y, press, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# 添加将值映射到颜色的颜色栏
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.add_axes(ax)
plt.show()