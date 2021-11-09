from xgboost import XGBRegressor as XGBR  # xgboost回归
from sklearn.ensemble import RandomForestRegressor as RFR  # 随机森林回归
from sklearn.linear_model import LinearRegression as LinearR  # 线性回归
from sklearn.ensemble import GradientBoostingRegressor  # GBDT

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS  # K则交叉验证
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from time import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import svm
import read_file
from sklearn.tree import DecisionTreeRegressor


def produce_data():
    wine = datasets.load_wine()  # 178*13
    bot = datasets.load_boston()  # 506*13
    digits = datasets.load_digits()  # 1797*64

    # X, y = read_file.read_file_airfoil_self_noise()  # 1503*5
    X, y = read_file.read_file_Folds5x2_pp()  # 9568*4
    # X = digits.data
    # print(X.shape)
    # y = digits.target

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    # X = X_scaler.fit_transform(X)
    # y = y_scaler.fit_transform(y.reshape(-1, 1))

    X = X_scaler.fit_transform(list(map(list, zip(*X))))  # list装置
    y = y_scaler.fit_transform(np.array(y).reshape(-1, 1))
    return X, y


# 求模型MSE
def MSE(y_true, y_predict):
    mse = mean_squared_error(y_true, y_predict)

    return mse


# 求模型MSE
def MAE(y_true, y_predict):
    mae = mean_absolute_error(y_true, y_predict)

    return mae


# 求模型R2
def R2(y_true, y_predict):
    # print(len(y_predict))
    r2 = r2_score(y_predict, y_true)
    return r2


def XGBRegressor1():
    # axisx = range(100, 101, 1)
    # MSE_rs = []
    # MAE_rs = []
    # r2_rs = []
    # for i in axisx:
    #     reg = XGBR(n_estimators=i).fit(X_train, y_train)
    #     y_predict = reg.predict(X_test)  # 传统接口predict
    #     mse = MSE(y_test, y_predict)
    #     mae = MAE(y_test, y_predict)
    #     r2 = R2(y_test, y_predict)
    #     print("%s棵树xgboost损失:" % i, mse, mae, r2)
    #     MSE_rs.append(mse)
    #     MAE_rs.append(mae)
    #     r2_rs.append(r2)
    # print(axisx[MSE_rs.index(min(MSE_rs))], min(MSE_rs))
    # print(axisx[MAE_rs.index(min(MAE_rs))], min(MAE_rs))
    # print(axisx[r2_rs.index(min(r2_rs))], min(r2_rs))
    # plt.figure(figsize=(12, 5))
    # plt.plot(axisx, MSE_rs, c="red", label="XGBRegressor")
    # plt.legend()
    # plt.xlabel('tree_estimators')
    # plt.ylabel('MSE')
    # plt.show()
    reg = XGBR(n_estimators=20).fit(X_train, y_train)
    y_predict = reg.predict(X_test)  # 传统接口predict
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("XGBR_MSE：", mse)
    print("XGBR_MAE：", mae)
    print("XGBR_R2：", r2)


def RandomForestRegressor1():
    axisx = range(1, 501, 20)
    MSE_rs = []
    MAE_rs = []
    r2_rs = []
    for i in axisx:
        reg = RFR(n_estimators=i).fit(X_train, y_train)
        y_predict = reg.predict(X_test)  # 传统接口predict
        mse = MSE(y_test, y_predict)
        mae = MAE(y_test, y_predict)
        r2 = R2(y_test, y_predict)
        print("%s棵树xgboost损失:" % i, mse, mae, r2)
        MSE_rs.append(mse)
        MAE_rs.append(mae)
        r2_rs.append(r2)
    print(axisx[MSE_rs.index(min(MSE_rs))], min(MSE_rs))
    print(axisx[MAE_rs.index(min(MAE_rs))], min(MAE_rs))
    print(axisx[r2_rs.index(min(r2_rs))], min(r2_rs))
    plt.figure(figsize=(12, 5))
    plt.plot(axisx, MSE_rs, c="red", label="RandomForestRegressor")
    plt.legend()
    plt.xlabel('tree_estimators')
    plt.ylabel('MSE')
    plt.show()


# 线性回归
def LinearRegression1():
    reg = LinearR().fit(X_train, y_train)
    y_predict = reg.predict(X_test)  # 传统接口predict
    # print(y_predict)
    # print(y_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("LR_MSE：", mse)
    print("LR_MAE：", mae)
    print("LR_R2：", r2)


# GBDT回归
def GBDTRegression():
    reg = GradientBoostingRegressor().fit(X_train, y_train)
    y_predict = reg.predict(X_test)  # 传统接口predict
    # print(y_predict)
    # print(y_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("GBDTR_MSE：", mse)
    print("GBDTR_MAE：", mae)
    print("GBDTR_R2：", r2)


# 构建lasso
def lassoRegression():
    lasso = LassoCV(alphas=np.logspace(-3, 1, 20))
    lasso.fit(X_train, y_train)
    y_predict = lasso.predict(X_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("lassoR_MSE：", mse)
    print("lassoR_MAE：", mae)
    print("lassoR_R2：", r2)


# 构建岭回归
def ridgeRegression():
    # 构建岭回归
    ridge = RidgeCV(alphas=np.logspace(-3, 1, 20))
    ridge.fit(X_train, y_train)
    y_predict = ridge.predict(X_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("ridgeR_MSE：", mse)
    print("ridgeR_MAE：", mae)
    print("ridgeR_R2：", r2)


# 回归树
def DTRegressor():
    # 构建模型（回归）
    model = DecisionTreeRegressor(criterion='mse', max_depth=10)
    # 模型训练
    model.fit(X_train, y_train)
    # 模型预测
    y_predict = model.predict(X_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("DTR_MSE：", mse)
    print("DTR_MAE：", mae)
    print("DTR_R2：", r2)


def svm_svc(X_train, X_test, y_train, y_test):
    clf = svm.SVR(kernel='rbf')
    clf.fit(X_train, y_train)
    new_prediction = clf.predict(X_test)
    mse = MSE(y_test, new_prediction)
    mae = MAE(y_test, new_prediction)
    r2 = R2(y_test, new_prediction)
    print("SVR_MSE：", mse)
    print("SVR_MAE：", mae)
    print("SVR_R2：", r2)


if __name__ == '__main__':
    time0 = time()
    X, y = produce_data()

    # 样本数据
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    XGBRegressor1()
    GBDTRegression()
    # RandomForestRegressor1()
    # LinearRegression1()
    # lassoRegression()
    # ridgeRegression()
    # DTRegressor()
    # svm_svc(X_train, X_test, y_train, y_test)
    print("模型运行时间：", time() - time0)
