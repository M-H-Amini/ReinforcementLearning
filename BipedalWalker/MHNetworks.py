from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt

def createModel(input_dim, hidden_units, output_dim=1):
    model = Sequential()
    model.add(Dense(hidden_units, 'tanh', input_shape=(input_dim,)))
    model.add(Dense(output_dim))
    return model 


