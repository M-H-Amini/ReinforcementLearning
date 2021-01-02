import numpy as np
import matplotlib.pyplot as plt

eps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0]
files = ['Cartpole-eps{}.npy'.format(i) for i in eps[:-1]]
files.append('FuzzyCartpole.npy')
print(files)
data = [np.load(file) for file in files]

def plotDataMean(data, title):
    mean = np.mean(data, axis=0)
    plt.figure()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Mean Duration')
    plt.plot([i + 1 for i in range(len(mean))], mean, 'r--o')
    plt.show()

def plotAllDataMean(data_list, title, indexes, eps):
    mean = [np.mean(data_list[i], axis=0) for i in range(len(data_list))]
    plt.figure()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Mean Duration')
    for i in range(len(mean)):
        plt.plot([j + 1 for j in range(len(mean[i]))], mean[i], '--o', label='eps={}'.format(eps[indexes[i]]))
    plt.legend()
    plt.show()


def showData(data):
    print(data)

def showStatistics(data, eps):
    mean_episode_duration = []
    for i in range(len(data)):
        mean_episode_duration.append(np.mean(np.mean(data[i])))
        if i < len(data) - 1:
            print('Eps = {} , Mean Duration = {}'.format(eps[i], mean_episode_duration[i]))
        else:
            print('Eps = Fuzzy, Mean Duration = {}'.format(mean_episode_duration[-1]))
    plt.figure()
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Duration')
    plt.plot(eps, mean_episode_duration, 'gx')
    plt.show()


showData(data[-1])
showStatistics(data, eps)
plotDataMean(data[0], r'eps={}'.format(eps[0]))
indexes = [0, 1,-3, -2, -1]
plotAllDataMean([data[i] for i in indexes], 'Durations for different epsilons', indexes, [eps[i] for i in indexes])
eps_history = np.load('eps.npy')
#print(eps)