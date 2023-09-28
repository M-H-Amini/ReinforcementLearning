#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                   Title: Q-Network                                  ##
##                                   Date: 2023/09/10                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import numpy as np
import logging as log
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.optimizers import SGD

def isForbidden(s):
    return s in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

def s2x(s):
    x = np.zeros((18,))
    ns = [isForbidden(t) for t in [s - 12, s + 1, s + 12, s - 1]]
    x[-4:] = ns
    r = s // 12
    c = s % 12
    x[r] = r
    x[c + 4] = c
    return x[np.newaxis, :]

def generateQNetworks():
    inputs = Input(shape=(18,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    actions = [Dense(1, activation='linear')(x) for _ in range(4)]
    models = [Model(inputs=inputs, outputs=actions[a]) for a in range(4)]
    for model in models:
        model.compile(loss='mse', optimizer='adam')
    return models


def generateZeroModel(model):
    zero_model = clone_model(model)
    for w in zero_model.trainable_variables:
        w.assign(np.zeros(w.shape))
    return zero_model

def redundantTrain(models, input_shape=(18,), n=10):
    print('here we are...')
    for model in models:
        for i in range(n):
            X = np.array([s2x(np.random.randint(37))[0] for _ in range(32)])
            y = np.ones((32, 4)) * 100
            model.fit(X, y, epochs=10, verbose=0)
    return models


def updateModels(models, X, y):
    print('Training models...', end='')
    print([len(x) for x in X], end='')
    losses = [-1 for _ in range(4)]
    for i, model in enumerate(models):
        if len(X[i]):
            hist = model.fit(X[i], y[i], epochs=5, verbose=0)
            losses[i] = round(hist.history['loss'][-1], 4)
    print(losses, end='')
    print('\r', end='')


    return models
    
        

if __name__ == '__main__':
    x = np.random.rand(1, 16)
    models = generateQNetworks()
    
    

