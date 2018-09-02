import rbm_lib as rbm
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("dataset/ECG5000/ECG5000_TRAIN", header = None)
dataset = dataset.drop(dataset.columns[0], axis=1)
data = dataset.values
num_hidden_units = 120
epoch = 5000
activation = "LeakyReLU"
learning_rate = 0.1

weights, params = rbm.trainRBM(data, num_hidden_units, epoch, activation, learning_rate)




dataset = pd.read_csv("dataset/ECG5000/ECG5000_TEST", header = None)
dataset = dataset.drop(dataset.columns[0], axis=1)
data = dataset.values

reconstructed_data = rbm.reconstructData(data, weights, params)

rmse = sqrt(mean_squared_error(data, reconstructed_data))
print(rmse)