import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def ReLU(data):
    data[data < 0] = 0
    return data

def LeakyReLU(data):
    data = np.array(data, dtype= float)
    for (x,y), value in np.ndenumerate(data):
        if data[x,y] < 0:
            data[x,y] = (data[x,y] * 0.01)


    return data


def trainRBM(data, num_hidden_units, epoch, activation, learning_rate):
    original_data = data
    weight_init = "random"
    sc = MinMaxScaler()
    data = sc.fit_transform(data)
    params = [num_hidden_units, activation, learning_rate]

    visible_to_hidden = np.empty([2, 2])


    num_instances = data.shape[0]

    weights = np.full((data.shape[1], num_hidden_units), 0.1)


    for counter in range(epoch):

        #Weighted input vectors passed through Rectified Linear Units
        if activation == 'relu':
            visible_to_hidden = ReLU(np.dot(data, weights))

        elif activation == 'LeakyReLU':
            visible_to_hidden = LeakyReLU(np.dot(data, weights))

        else:
            visible_to_hidden = LeakyReLU(np.dot(data, weights))

        #Sigmoid function for probabilistic hidden units activation
        sigma_visible_to_hidden = sigmoid(visible_to_hidden)

        #Asses if hidden units are activated according to > some random weights
        #The weight assesment needs to be restructured
        hidden_states = sigma_visible_to_hidden > np.random.rand(num_instances, num_hidden_units)

        visible_associations = np.dot(data.T, hidden_states) #xi * xj where i denotes visible units and j denotes hidden units


        #Hidden layer to visible layer reconstruction
        hidden_to_visible = np.dot(sigma_visible_to_hidden, weights.T)
        #Sigmoid function applied to visible layer
        sigma_hidden_to_visible = sigmoid(hidden_to_visible)

        #Visible to hidden again (We only apply ReLU or LeakyReLU in feedforward manner)
        if activation == 'relu':
            hidden_activations = ReLU(np.dot(sigma_hidden_to_visible, weights))

        elif activation == 'LeakyReLU':
            hidden_activations = LeakyReLU(np.dot(sigma_hidden_to_visible, weights))

        else:
            hidden_activations = LeakyReLU(np.dot(sigma_hidden_to_visible, weights))

        #Sigmoid
        sigma_hidden_activation = sigmoid(hidden_activations)

        #sigma_hidden_to_visible represents the reconstructed output this time
        hidden_associations = np.dot(sigma_hidden_to_visible.T, sigma_hidden_activation)

        #Update weights using Contrastive Divergence
        weights += learning_rate * ((visible_associations - hidden_associations) / num_instances)

        #Reconstruct data to calculate RMSE loss
        reconstructed_data = sc.inverse_transform(sigma_hidden_to_visible)


        rmse = sqrt(mean_squared_error(original_data, reconstructed_data))
        print("Epoch: "+str(counter+1)+" loss is "+str(rmse))

    return weights, params

def reconstructData(data, weights, params):
    sc = MinMaxScaler()
    data = sc.fit_transform(data)

    activation = params[1]


    visible_to_hidden = 0

    #Weighted input vectors passed through Rectified Linear Units
    if activation == 'relu':
        visible_to_hidden = ReLU(np.dot(data, weights))

    elif activation == 'LeakyReLU':
        visible_to_hidden = LeakyReLU(np.dot(data, weights))

    else:
        visible_to_hidden = LeakyReLU(np.dot(data, weights))

    #Sigmoid function for probabilistic hidden units activation
    sigma_visible_to_hidden = sigmoid(visible_to_hidden)


    #Hidden layer to visible layer reconstruction
    hidden_to_visible = np.dot(sigma_visible_to_hidden, weights.T)
    #Sigmoid function applied to visible layer
    sigma_hidden_to_visible = sigmoid(hidden_to_visible)

    #Reconstruct data to calculate RMSE loss
    reconstructed_data = sc.inverse_transform(sigma_hidden_to_visible)

    return reconstructed_data
