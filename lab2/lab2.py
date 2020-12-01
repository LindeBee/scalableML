import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

def load_data(data_path = "dataF8k/captions.txt"):
    return pd.read_csv(data_path, header = 0)

data = load_data()
data.head()

# def f(x):
#     return np.sin(x)

# xTrain = np.arange(0, 2*np.pi, 0.01)
# yTrain = f(xTrain)
# xTest = np.arange(0.05, 2*np.pi, 0.1)
# yTest = f(xTest)

# fig = plt.subplot()
# fig.scatter(xTrain,yTrain)
# plt.show()

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(100, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(49, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(1))
# model.compile(loss='mean_squared_error')

# model.fit(xTrain, yTrain, batch_size = 100, epochs = 300, shuffle = True)
# pTest = model.predict(xTest)
# pTrain = model.predict(xTrain)

# fig = plt.subplot()
# fig.scatter(xTest,pTest)
# fig.scatter(xTest,yTest)
# fig.scatter(xTrain,pTrain)
# plt.show()

