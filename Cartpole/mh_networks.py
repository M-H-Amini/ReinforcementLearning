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
