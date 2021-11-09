from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
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
import tf_sklearn_network
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeRegressor


def produce_data():
    wine = datasets.load_wine()  # 178*13
    bot = datasets.load_boston()  # 506*13
    digits = datasets.load_digits()  # 1797*64

    # X, y = read_file.read_file_airfoil_self_noise()  # 1503*5
    X, y = read_file.read_file_Folds5x2_pp()  # 9568*4
    # X = wine.data
    # print(X.shape)
    # y = wine.target

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


# 神经网络回归
def BPRegression(X_train, y_train, X_test, y_test, network_epochs, hidden_unit_number, hidden_layer_num,
                 regularization_learn_rate):
    model = Sequential()
    feature_number = X_train.shape[1]

    label_number = y_train.shape[1]

    model.add(Dense(units=hidden_unit_number, activation='sigmoid', input_dim=feature_number,
                    activity_regularizer=regularizers.l2(regularization_learn_rate)))
    for i in range(hidden_layer_num):
        model.add(Dense(units=hidden_unit_number, activation='sigmoid',
                        activity_regularizer=regularizers.l2(regularization_learn_rate)))
    model.add(Dense(units=label_number, activation='sigmoid',
                    activity_regularizer=regularizers.l2(regularization_learn_rate)))

    model.compile(loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
    history = model.fit(X_train, y_train, epochs=network_epochs, callbacks=[early_stopping], verbose=2)

    y_predict = model.predict(X_test)
    mse = MSE(y_test, y_predict)
    mae = MAE(y_test, y_predict)
    r2 = R2(y_test, y_predict)
    print("BPR_MSE：", mse)
    print("BPR_MAE：", mae)
    print("BPR_R2：", r2)
    return mse


if __name__ == '__main__':
    time0 = time()
    history_bp_mse = []  # 存储mse
    last_bp_mse = 0  # 上一次mse
    now_bp_mse = 0  # 这一次mse
    history_bp_mse_lost = []  # 存储梯度
    last_bpboosting_mse_lost = 0
    now_bpboosting_mse_lost = 0
    history_bpboosting_mse = []
    history_bpboosting_mse_lost = []
    network_max_epochs = 200
    model_max_epochs = 5

    network_number = 10
    hidden_layer_num = 5
    regularization_learn_rate = 0.03
    hidden_unit_number = 20
    axisx = range(10, 510, 20)
    axisx_bp = range(10 * model_max_epochs, 510 * model_max_epochs, 20 * model_max_epochs)
    # 样本数据
    X, y = produce_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    layer_axisx = range(1, 21, 1)
    for i in layer_axisx:
        # 神经网络
        now_bp_mse = BPRegression(X_train, y_train, X_test, y_test, network_max_epochs * model_max_epochs,
                                  hidden_unit_number, i, regularization_learn_rate)
        history_bp_mse.append(now_bp_mse)

        # 神经网络Boosting
        train_model = tf_sklearn_network.boost_train(model_max_epochs, network_number, X_train, y_train,
                                                     network_max_epochs,
                                                     hidden_unit_number, i, regularization_learn_rate)
        now_bpboosting_mse = tf_sklearn_network.boost_predict(model_max_epochs, network_number, X_test, y_test,
                                                              network_max_epochs, hidden_unit_number, i,
                                                              regularization_learn_rate, train_model)
        history_bpboosting_mse.append(now_bpboosting_mse)

    # min_mse = min(history_bp_mse)
    plt.figure(figsize=(12, 5))

    plt.plot(layer_axisx, history_bp_mse, c="blue", label="netWork", linestyle='-', marker='v')
    plt.plot(layer_axisx, history_bpboosting_mse, c="red", label="NNBoost", linestyle='--', marker='o')
    # plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()

    plt.xlabel('hidden_layer_num')
    plt.ylabel('MSE')
    plt.show()
    # svm_svc(X_train, X_test, y_train, y_test)
    print("模型运行时间：", time() - time0)
