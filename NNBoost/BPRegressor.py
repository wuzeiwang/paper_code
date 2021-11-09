from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import datasets

(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()

x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))
print('-----------------')
print(y_train_pd.head(5))

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

model = Sequential()
print(x_train_pd.shape)
model.add(Dense(units=10, activation='relu', input_dim=x_train_pd.shape[1]))

model.add(Dropout(0.2))
# 权重正则化，输出正则化
model.add(Dense(units=15, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
history = model.fit(x_train, y_train, epochs=20, batch_size=200, callbacks=[early_stopping], verbose=2,
                    validation_data=(x_valid, y_valid))
# print(model.summary())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

y_new = model.predict(x_valid)
for i in range(len(y_new)):
    print(y_new[i], y_valid[i])
# print(y_valid)
# print('-----------------')
# print(y_new)

min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)


def netWork_start(X_train, y_train, network_epochs, hidden_unit_number, hidden_layer_num, regularization_learn_rate):
    model = Sequential()

    model.add(Dense(units=10, activation='sigmoid'))

    # 权重正则化，输出正则化
    # model.add(Dense(units=15, activation='sigmoid'))
    for i in range(hidden_layer_num):
        model.add(Dense(units=hidden_unit_number, activation='sigmoid'
                        ))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam')

    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
    history = model.fit(X_train, y_train, epochs=200, callbacks=[early_stopping], verbose=2)
    y_new = model.predict(X_train)
    for i in range(len(y_new)):
        print(y_new[i], y_train[i])

    return model


def netWork_start1(X_train, y_train, network_epochs, hidden_unit_number, hidden_layer_num, regularization_learn_rate):
    model = Sequential()
    feature_number = X_train.shape[1]
    label_number = y_train.shape[1]
    model.add(Dense(units=10, activation='sigmoid'
                    ))
    for i in range(hidden_layer_num):
        model.add(Dense(units=hidden_unit_number, activation='sigmoid'
                        ))
    model.add(Dense(units=label_number, activation='sigmoid'
                    ))
    # model.build((None, feature_number))

    model.compile(loss='mse', optimizer='adam')
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min')
    history = model.fit(X_train, y_train, epochs=network_epochs, callbacks=[early_stopping], verbose=2)
    # model.summary()  # 模型预览
    print('-----------------')
    y_predict = model.predict(X_train)
    for i in range(len(y_train)):
        print(y_predict[i], y_train[i])

    return y_predict, model


if __name__ == '__main__':
    # 网络参数
    network_max_epochs = 200
    model_max_epochs = 1

    network_number = 2
    hidden_layer_num = 5
    regularization_learn_rate = 0.03
    hidden_unit_number = 20

    boston = datasets.load_boston()  # 506*13

    X = boston.data
    y = boston.target
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    # 样本数据
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # netWork_start(X_train, y_train, network_max_epochs, hidden_unit_number, hidden_layer_num, regularization_learn_rate)
    netWork_start1(X_train, y_train, network_max_epochs, hidden_unit_number, hidden_layer_num,
                   regularization_learn_rate)
