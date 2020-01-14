import numpy as np
import matplotlib.pyplot as plt

data = np.load('Cartpole.npy')
print(data.shape)
def plotDataMean(data):
    mean = np.mean(data, axis=0)
    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(i + 1, mean[i], 'ro')
    plt.show()

plotDataMean(data)