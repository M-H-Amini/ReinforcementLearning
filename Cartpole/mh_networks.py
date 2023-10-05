import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.optimizers import SGD
import numpy as np

def createCritic(input_shape=(4,)):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def createActor(input_shape=(4,), actions=2):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    if actions == 2:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(actions, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def initialTrain(model):
    X, y = generateRandomData(10000)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=10, verbose=1)
    return model

def generateRandomData(n=1000, p=0.75):
    X, y = [], []
    for i in range(n):
        x = np.random.rand() * 9.6 - 4.8
        x_dot = np.random.rand() * 5 - 2.5
        theta = np.random.rand() * 0.4 - 0.2
        theta_dot = np.random.rand() * 5 - 2.5
        X.append([x, x_dot, theta, theta_dot])
        target = p if theta > 0 else 1 - p
        y.append(target)
    return np.array(X), np.array(y)
    
if __name__ == '__main__':
    model = createActor()
    model = initialTrain(model)
    