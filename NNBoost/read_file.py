# 相关库和函数导入
import numpy as np
import pandas as pd
import urllib
import scipy as sp
import random
import matplotlib.pyplot as plt  # 导入常用的基本库


# 1503*5
def read_file_airfoil_self_noise():
    # 数据导入
    # 导入数据，注意编码使用gbk，然后含有中文字符的地址要设定"engine=python"
    # 要自己去UCI网站下载数据
    dt = pd.read_table("F:/Data/Regression/airfoil_self_noise/airfoil_self_noise.dat", encoding='gbk', engine='python',
                       sep="\s+", header=None)
    # display(dt.duplicated().any())
    # print(dt[0][0], dt[0][1])
    # print(len(dt.values), len(dt.values[0]))
    # dt.drop_duplicates(inplace=True)  # 查看数据集的数据，注意下标从0开始
    X = [[0 for i in range(len(dt.values))] for j in range(len(dt.values[0]) - 1)]
    # print(len(X))
    y = [[0 for i in range(len(dt.values))] for j in range(1)]
    # print(len(y))
    # 存储属性
    for i in range(len(dt.values[0]) - 1):
        for j in range(len(dt.values)):
            X[i][j] = dt[i][j]
    # 存储标签
    for j in range(len(dt.values)):
        y[0][j] = dt[(len(dt.values[0]) - 1)][j]

    print(len(X[0]), len(y[0]))
    return X, y


# 9568*4
def read_file_Folds5x2_pp():
    # Folds5x2_pp数据导入
    # 导入数据，注意编码使用gbk，然后含有中文字符的地址要设定"engine=python"
    # 要自己去UCI网站下载数据
    dt = pd.read_excel("F:/Data/Regression/Combined_Cycle_Power_Plant/CCPP/Folds5x2_pp.xlsx")

    X = [[0 for i in range(len(dt.values))] for j in range(len(dt.values[0]) - 1)]

    y = [[0 for i in range(len(dt.values))] for j in range(1)]
    # print(len(y))
    # 存储特征

    for j in range(len(dt.values[0]) - 1):
        for i in range(len(dt.values)):
            # print(dt.values[i][j])
            X[j][i] = dt.values[i][j]
    # 存储标签
    for j in range(len(dt.values)):
        y[0][j] = dt.values[j][(len(dt.values[0]) - 1)]

    print(len(X[0]), len(y[0]))

    return X, y


# 1503*5
def read_file_Gas_sensor_array_under_dynamic_gas_mixtures():
    # 数据导入
    # 导入数据，注意编码使用gbk，然后含有中文字符的地址要设定"engine=python"
    dt = pd.read_table("F:/Data/Regression/Gas_sensor_array_under_dynamic_gas_mixtures_Data_Set/ethylene_CO.txt",
                       encoding='gbk', engine='python', sep="\s+", header=None)
    # display(dt.duplicated().any())
    # print(dt[0][0], dt[0][1])
    print(len(dt.values), len(dt.values[0]))
    X = [[0 for i in range(len(dt.values))] for j in range(len(dt.values[0]) - 1)]
    # print(len(X))
    y = [[0 for i in range(len(dt.values))] for j in range(1)]
    # print(len(y))
    # 存储属性
    for i in range(len(dt.values[0]) - 1):
        for j in range(len(dt.values)):
            X[i][j] = dt[i][j]
    # 存储标签
    for j in range(len(dt.values)):
        y[0][j] = dt[(len(dt.values[0]) - 1)][j]

    print(len(X[0]), len(y[0]))
    return X, y


if __name__ == '__main__':
    # 实验用的是前两个数据集，在论文中为D4, D5
    # read_file_airfoil_self_noise()
    # read_file_Folds5x2_pp()
    read_file_Gas_sensor_array_under_dynamic_gas_mixtures()
