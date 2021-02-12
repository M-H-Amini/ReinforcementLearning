import numpy as np
import matplotlib.pyplot as plt

data = np.load('nstep.npy')
show_number = 3

print(data.shape)

plt.figure()
indexes = np.random.randint(0, len(data), show_number)
for i in indexes:
    plt.plot(range(len(data[i])), data[i], '-x')
plt.show()

data_done = data==200
episodes_done_mean = np.sum(data_done, axis=1)
mean_done = np.mean(episodes_done_mean)
std_done = np.std(episodes_done_mean)
print(mean_done)
print(std_done)
