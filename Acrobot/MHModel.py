import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def buildModel():
    model = Sequential([
        Dense(5, 'tanh', input_shape=(6,)),
        Dense(3, 'tanh'),
        Dense(3)
    ])
    return model

def processModel(model, i):
    w = np.zeros((3, 1))
    w[i, 0] = 1
    model.layers[-1].set_weights([w, np.array([0])])
    return model