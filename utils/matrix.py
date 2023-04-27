import numpy as np

def up_triangle_matrix(n,m):
    if m==0:
        matrix = np.zeros((n, n))  # 创建一个n x n的全零矩阵
        for i in range(n):
            for j in range(i, n):
                matrix[i][j] = 1  # 上三角部分赋值为1
        return matrix
    else:
        matrix = np.zeros((n, n))  # 创建一个n x n的全零矩阵
        for i in range(n):
            if i+m+1<n:
                for j in range(i, i+m+1):
                    matrix[i][j] = 1  # 上三角部分赋值为1
            else:
                for j in range(i, n):
                    matrix[i][j] = 1  # 上三角部分赋值为1
        return matrix
def up_cling_matrix(n,m,value):
    matrix = np.zeros((n, n))  # 创建一个n x n的全零矩阵
    for i in range(n):
        if i+m<n:
            matrix[i][i+m] = value  # 上三角部分赋值为1
    return matrix
# def up_cling_matrix_dengcha:


if __name__ == "__main__":
    a=up_cling_matrix(7,2,3)
    print(a)