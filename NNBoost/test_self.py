import tf_sklearn_network
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import read_file
import numpy as np
from time import time


# 测试神经网络个数对训练结果的影响
def test_network_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs, hidden_unit_number,
                     hidden_layer_num, regularization_learn_rate):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(model_max_epochs, i, X_train, y_train, network_max_epochs,
                                                     hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(tf_sklearn_network.boost_predict(model_max_epochs, i, X_test, y_test, network_max_epochs,
                                                            hidden_unit_number, hidden_layer_num,
                                                            regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


# 测试隐藏层单元个数对训练结果的影响
def test_hidden_unit_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs, network_number,
                         hidden_layer_num, regularization_learn_rate):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(model_max_epochs, network_number, X_train, y_train,
                                                     network_max_epochs,
                                                     i, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            tf_sklearn_network.boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs,
                                             i, hidden_layer_num, regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


# 测试隐藏层数对训练结果的影响
def test_hidden_layer_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs, network_number,
                          hidden_unit_number, regularization_learn_rate):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(model_max_epochs, network_number, X_train, y_train,
                                                     network_max_epochs, hidden_unit_number, i,
                                                     regularization_learn_rate)
        history_mse.append(
            tf_sklearn_network.boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs,
                                             hidden_unit_number, i, regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


# 测试正则化率对训练结果的影响
def test_regularization_learn_rate(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs,
                                   network_number, hidden_unit_number, hidden_layer_num):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(model_max_epochs, network_number, X_train, y_train,
                                                     network_max_epochs,
                                                     hidden_unit_number, hidden_layer_num, i / 100)
        history_mse.append(
            tf_sklearn_network.boost_predict(model_max_epochs, network_number, X_test, y_test, network_max_epochs,
                                             hidden_unit_number, hidden_layer_num, i / 100, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


# 测试模型迭代次数对训练结果的影响
def test_model_max_epochs(axisx, X_train, y_train, X_test, y_test, network_max_epochs, network_number,
                          hidden_unit_number, hidden_layer_num, regularization_learn_rate):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(i, network_number, X_train, y_train, network_max_epochs,
                                                     hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            tf_sklearn_network.boost_predict(i, network_number, X_test, y_test, network_max_epochs, hidden_unit_number,
                                             hidden_layer_num, regularization_learn_rate, train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


# 测试神经网络迭代次数对训练结果的影响
def test_network_max_epochs(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_number,
                            hidden_unit_number, hidden_layer_num, regularization_learn_rate):
    history_mse = []

    for i in axisx:
        train_model = tf_sklearn_network.boost_train(model_max_epochs, network_number, X_train, y_train, i,
                                                     hidden_unit_number, hidden_layer_num, regularization_learn_rate)
        history_mse.append(
            tf_sklearn_network.boost_predict(model_max_epochs, network_number, X_test, y_test, i,
                                             hidden_unit_number, hidden_layer_num, regularization_learn_rate,
                                             train_model))
    min_mse = min(history_mse)
    print(axisx[history_mse.index(min_mse)] + 1, min_mse)
    return history_mse


def draw_lines(axisx, result, data_label, xlabel):
    N = len(result)
    plt.figure(figsize=(12, 6))
    color = ["blue", "cyan", "green", "black", "magenta", "red", "white", "yellow"]
    linestyle = [':', '-', '--', '-.', ':']
    maker = ['.', ',', 'o', 'v', '<']
    for i in range(N):
        plt.plot(axisx, result[i], c=color[i], linestyle=linestyle[i], label=data_label[i], marker=maker[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('MSE')
    plt.show()
    # 显示5秒后自动关闭
    # plt.ion()
    # plt.pause(2)  # 显示秒数
    # plt.close()


def load_data():
    X = []
    y = []
    boston = datasets.load_boston()  # 506*13
    digits = datasets.load_digits()  # 1797*64
    wine = datasets.load_wine()  # 178*13
    airfoil_self_noise_data, airfoil_self_noise_target = read_file.read_file_airfoil_self_noise()  # 1503*5
    Folds5x2_pp_data, Folds5x2_pp_target = read_file.read_file_Folds5x2_pp()  # 9568*4
    X.append(boston.data)
    y.append(boston.target)
    X.append(digits.data)
    y.append(digits.target)
    X.append(wine.data)
    y.append(wine.target)

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    airfoil_self_noise_data = X_scaler.fit_transform(list(map(list, zip(*airfoil_self_noise_data))))  # list装置
    airfoil_self_noise_target = y_scaler.fit_transform(np.array(airfoil_self_noise_target).reshape(-1, 1))
    Folds5x2_pp_data = X_scaler.fit_transform(list(map(list, zip(*Folds5x2_pp_data))))  # list装置
    Folds5x2_pp_target = y_scaler.fit_transform(np.array(Folds5x2_pp_target).reshape(-1, 1))
    for i in range(len(X)):
        X[i] = X_scaler.fit_transform(X[i])
        y[i] = y_scaler.fit_transform(y[i].reshape(-1, 1))
    X.append(airfoil_self_noise_data)
    y.append(airfoil_self_noise_target)
    X.append(Folds5x2_pp_data)
    y.append(Folds5x2_pp_target)
    return X, y


def main():
    # 网络参数
    network_max_epochs = 100
    model_max_epochs = 3

    network_number = 3
    hidden_layer_num = 5
    regularization_learn_rate = 0.02
    hidden_unit_number = 20

    X, y = load_data()
    axisx = range(1, 31, 5)
    netWork_num_result = []
    netWork_epoch_num_result = []
    model_epoch_num_result = []
    hidden_unit_num_result = []
    hidden_layer_num_result = []
    regularization_learn_rate_result = []

    label = ["boston", "digits", "wine", "airfoil_self_noise", "Folds5x2_pp"]

    for i in range(len(X)):
        print(X[i].shape, y[i].shape)
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i])
        # netWork_num_result.append(
        #     test_network_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs,
        #                      hidden_unit_number, hidden_layer_num, regularization_learn_rate))
        # netWork_epoch_num_result.append(
        #     test_network_max_epochs(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_number,
        #                             hidden_unit_number, hidden_layer_num, regularization_learn_rate))
        # hidden_unit_num_result.append(
        #     test_hidden_unit_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs,
        #                          network_number, hidden_layer_num, regularization_learn_rate))
        # hidden_layer_num_result.append(
        #     test_hidden_layer_num(axisx, model_max_epochs, X_train, y_train, X_test, y_test, network_max_epochs,
        #                           network_number, hidden_unit_number, regularization_learn_rate))
        # regularization_learn_rate_result.append(
        #     test_regularization_learn_rate(axisx, model_max_epochs, X_train, y_train, X_test, y_test,
        #                                    network_max_epochs, network_number, hidden_unit_number, hidden_layer_num))
        model_epoch_num_result.append(
            test_model_max_epochs(axisx, X_train, y_train, X_test, y_test, network_max_epochs, network_number,
                                  hidden_unit_number, hidden_layer_num, regularization_learn_rate))
    # draw_lines(axisx, netWork_num_result, label, "netWork_num")
    draw_lines(axisx, model_epoch_num_result, label, "model_epoch_num")


if __name__ == '__main__':
    mse_final = 6.5e-4
    time0 = time()
    main()
    print("模型运行时间：", time() - time0)
