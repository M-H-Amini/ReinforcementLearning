import numpy as np
import matplotlib.pyplot as plt

eps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
files = ['Cartpole-eps{}.npy'.format(i) for i in eps]
print(files)
data = [np.load(file) for file in files]

def plotDataMean(data, title):
    mean = np.mean(data, axis=0)
    plt.figure()
    plt.title(title)
    plt.plot([i + 1 for i in range(len(mean))], mean, 'r--o')
    plt.show()

def showData(data):
    print(data)

def showStatistics(data, eps):
    mean_episode_duration = []
    for i in range(len(data)):
        mean_episode_duration.append(np.mean(np.mean(data[i])))
        print('Eps = {} , Mean Duration = {}'.format(eps[i], mean_episode_duration[i]))
    plt.figure()
    plt.plot(eps, mean_episode_duration, 'bx')
    plt.show()


showData(data[0])

showStatistics(data, eps)
#plotDataMean(data[0], 'eps = 0.1')
#plotDataMean(data[3], 'eps = 0.25')
#plotDataMean(data[-1], 'eps = 0.5')
