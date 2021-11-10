from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from time import time
import read_file
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mse_final = 6.5e-4

# 调用接口的神经网络进行实验，代码会更加简洁

def transpose(matrix):
    return zip(*matrix)


def produce_data():
    wine = datasets.load_wine()  # 178*13
    boston = datasets.load_boston()  # 506*13
    digits = datasets.load_digits()  # 1797*64

    # X = digits.data
    # # print(X)
    # y = digits.target
    X, y = read_file.read_file_airfoil_self_noise()  # 1503*5
    # X, y = read_file.read_file_Folds5x2_pp()  # 9568*4
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    # X = X_scaler.fit_transform(np.array(X).reshape(-1, 1))
    # y = y_scaler.fit_transform(y)

    # X = X_scaler.fit_transform(X)
    # y = y_scaler.fit_transform(y.reshape(-1, 1))

    X = X_scaler.fit_transform(list(map(list, zip(*X))))  # list装置
    y = y_scaler.fit_transform(np.array(y).reshape(-1, 1))
    return X, y


# 求模型MSE
def MSE(y_true, y_predict):
    mse_sum = 0
    mse_history = []
    for i in range(len(y_true)):
        mse = np.average(np.square(y_true[i] - y_predict[i]))
        mse_sum += mse
        mse_history.append(mse)
    # mse_test = np.sum((y_predict - y_true) ** 2) / len(y_true)
    mse_test = mean_squared_error(y_predict, y_true)
    # print(mse_history)
    return mse_test


# 求模型MAE
def MAE(y_true, y_predict):
    mae_sum = 0
    for i in range(len(y_true)):
        mae = np.average(abs(y_true[i] - y_predict[i]))
        mae_sum += mae
    # mae_test = np.sum(np.absolute(y_predict - y_true)) / len(y_true)
    mae_test = mean_absolute_error(y_predict, y_true)
    return mae_test


# 求模型R2
def R2(y_true, y_predict):
    R_Squared = 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
    r2 = r2_score(y_predict, y_true)
    return r2


def netWork_start(X_train, y_train, network_epochs, hidden_unit_number, hidden_layer_num, regularization_learn_rate):
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
    # model.build((None, feature_number))

    model.compile(loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
    history=model.fit(X_train, y_train, epochs=network_epochs, callbacks=[early_stopping], verbose=2)
    # model.summary()  # 模型预览
    # history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    y_predict = model.predict(X_train)
    # for i in range(len(y_train)):
    #     print(y_predict[i], y_train[i])

    return y_predict, model


def netWork_after(X_train, y_train, network_epochs, model):
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
    history = model.fit(X_train, y_train, epochs=network_epochs, callbacks=[early_stopping], verbose=2)
    y_predict = model.predict(X_train)

    return y_predict, model


def predict_step(X_test, model):
    y_predict = model.predict(X_test)
    # network.draw_loss_line(mse_history)
    return y_predict


def boost_train(model_max_epochs, network_number, X_train, y_train, network_epochs, hidden_unit_number,
                hidden_layer_num, regularization_learn_rate):
    train_model = [1] * network_number
    y = [1] * network_number

    y_before = 0
    # 初始化神经网络训练
    for i in range(network_number):
        y[i], train_model[i] = netWork_start(X_train, y_train - y_before, network_epochs, hidden_unit_number,
                                             hidden_layer_num, regularization_learn_rate)
        y_before += y[i]
    mse = MSE(y_train, y_before)
    print(mse)
    for i in range(model_max_epochs - 1):
        if mse < mse_final:
            print(mse)
            break
        else:
            y_before_then = 0
            for i in range(network_number):
                y[i], train_model[i] = netWork_after(X_train, y_train - y_before_then, network_epochs,
                                                     train_model[i])
                y_before_then += y[i]
            mse = MSE(y_train, y_before_then)
            # print(mse)

    return train_model


def boost_predict(model_max_epochs, network_num, X_test, y_test, network_epochs, hidden_unit_number,
                  hidden_layer_num, regularization_learn_rate, train_model):
    y_predict = [1] * network_num
    y_predict_sum = 0
    for i in range(network_num):
        y_predict[i] = predict_step(X_test, train_model[i])
        y_predict_sum += y_predict[i]

    mse = MSE(y_test, y_predict_sum)
    mae = MAE(y_test, y_predict_sum)
    r2 = R2(y_test, y_predict_sum)
    print(
        "%s个具有%s层%s个单元的隐藏层，迭代次数为%s次，正则化率为%s的神经网络boost模型运行%s次的MSE损失:" % (
            network_num, hidden_layer_num, hidden_unit_number,
            network_epochs, regularization_learn_rate,
            model_max_epochs), mse)
    print(
        "%s个具有%s层%s个单元的隐藏层，迭代次数为%s次，正则化率为%s的神经网络boost模型运行%s次的MAE损失:" % (
            network_num, hidden_layer_num, hidden_unit_number,
            network_epochs, regularization_learn_rate,
            model_max_epochs), mae)
    print(
        "%s个具有%s层%s个单元的隐藏层，迭代次数为%s次，正则化率为%s的神经网络boost模型运行%s次的R2损失:" % (
            network_num, hidden_layer_num, hidden_unit_number,
            network_epochs, regularization_learn_rate,
            model_max_epochs), r2)
    return mse


# 测试神经网络个数对训练结果的影响
def test_network_num(model_max_epochs, X_train, y_train, network_max_epochs, hidden_unit_number,
                     hidden_layer_num, regularization_learn_rate):
    history_mse = []
    axisx = range(1, 31, 1)

    for i in axisx:
        train_model = boost_train(model_max_epochs, i, X_train, y_train, network_max_epochs,
                                  hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(boost_predict(model_max_epochs, i, X_test, y_test, network_max_epochs, hidden_unit_number,
                                         hidden_layer_num, regularization_learn_rate, train_model))

    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)
    plt.figure(figsize=(12, 6))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('netWork_num')
    plt.ylabel('MSE')
    plt.show()


# 测试隐藏层单元个数对训练结果的影响
def test_hidden_unit_num(model_max_epochs, X_train, y_train, network_max_epochs, network_number,
                         hidden_layer_num, regularization_learn_rate):
    history_mse = []
    axisx = range(1, 101, 5)

    for i in axisx:
        train_model = boost_train(model_max_epochs, network_number, X_train, y_train, network_max_epochs,
                                  i, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs, i,
                          hidden_layer_num, regularization_learn_rate, train_model))

    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)
    plt.figure(figsize=(12, 5))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('hidden_unit_num')
    plt.ylabel('MSE')
    plt.show()


# 测试隐藏层层数对训练结果的影响
def test_hidden_layer_num(model_max_epochs, X_train, y_train, network_max_epochs, network_number,
                          hidden_unit_number, regularization_learn_rate):
    history_mse = []
    axisx = range(1, 40, 1)

    for i in axisx:
        train_model = boost_train(model_max_epochs, network_number, X_train, y_train, network_max_epochs,
                                  hidden_unit_number, i, regularization_learn_rate)
        history_mse.append(
            boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs, hidden_unit_number,
                          hidden_layer_num, regularization_learn_rate, train_model))

    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)
    plt.figure(figsize=(12, 5))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('hidden_layer_num')
    plt.ylabel('MSE')
    plt.show()


# 测试神经网络迭代次数对训练结果的影响
def test_network_max_epochs(model_max_epochs, X_train, y_train, network_number, hidden_unit_number,
                            hidden_layer_num, regularization_learn_rate):
    history_mse = []
    axisx = range(1, 301, 30)

    for i in axisx:
        train_model = boost_train(model_max_epochs, network_number, X_train, y_train, i,
                                  hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            boost_predict(model_max_epochs, network_number, X_test, y_test, i, hidden_unit_number,
                          hidden_layer_num, regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)

    plt.figure(figsize=(12, 5))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('network_epochs')
    plt.ylabel('MSE')
    plt.show()


# 测试模型迭代次数对训练结果的影响
def test_model_max_epochs(X_train, y_train, network_max_epochs, network_number, hidden_unit_number,
                          hidden_layer_num, regularization_learn_rate):
    history_mse = []
    axisx = range(1, 41, 4)

    for i in axisx:
        train_model = boost_train(i, network_number, X_train, y_train, network_max_epochs,
                                  hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            boost_predict(i, network_number, X_test, y_test, network_max_epochs, hidden_unit_number,
                          hidden_layer_num, regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)

    plt.figure(figsize=(12, 5))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('model_epochs')
    plt.ylabel('MSE')
    plt.show()


# 测试正则化率对训练结果的影响
def test_regularization_learn_rate(model_max_epochs, X_train, y_train, network_max_epochs, network_number,
                                   hidden_unit_number, hidden_layer_num):
    history_mse = []
    axisx = range(1, 101, 10)

    for i in axisx:
        print("第%s次", i)
        train_model = boost_train(model_max_epochs, network_number, X_train, y_train, network_max_epochs,
                                  hidden_unit_number, hidden_layer_num, i / 100)
        history_mse.append(
            boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs, hidden_unit_number,
                          hidden_layer_num, i / 100, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)], min_mse)

    plt.figure(figsize=(12, 5))
    plt.plot(axisx, history_mse, c="red", label="netWorkBoost")
    plt.plot([0, max(axisx)], [min_mse, min_mse])
    plt.legend()
    plt.xlabel('regularization_learn_rate * 100')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    time0 = time()
    # 网络参数
    network_max_epochs = 1200
    model_max_epochs = 10

    network_number = 5
    hidden_layer_num = 5
    regularization_learn_rate = 0.03
    hidden_unit_number = 40

    X, y = produce_data()
    # print(len(X[0]), len(y[0]))
    # 样本数据
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 神经网络个数
    # test_network_num(model_max_epochs, X_train, y_train, network_max_epochs, hidden_unit_number, hidden_layer_num,
    #                  regularization_learn_rate)
    # 隐藏层单元个数
    # test_hidden_unit_num(model_max_epochs, X_train, y_train, network_max_epochs, network_number,
    #                      hidden_layer_num, regularization_learn_rate)

    # 隐藏层层数
    # test_hidden_layer_num(model_max_epochs, X_train, y_train, network_max_epochs, network_number, hidden_unit_number,
    #                       regularization_learn_rate)

    # 神经网络迭代次数
    # test_network_max_epochs(model_max_epochs, X_train, y_train, network_number, hidden_unit_number,
    #                         hidden_layer_num, regularization_learn_rate)

    # 模型迭代次数
    # test_model_max_epochs(X_train, y_train, network_max_epochs, network_number, hidden_unit_number,
    #                       hidden_layer_num, regularization_learn_rate)

    # 正则化速率
    # test_regularization_learn_rate(model_max_epochs, X_train, y_train, network_max_epochs, network_number,
    #                                hidden_unit_number, hidden_layer_num)

    train_model = boost_train(model_max_epochs, network_number, X_train, y_train, network_max_epochs,
                              hidden_unit_number, hidden_layer_num, regularization_learn_rate)
    boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs, hidden_unit_number,
                  hidden_layer_num, regularization_learn_rate, train_model)
    print("模型运行时间：", time() - time0)
