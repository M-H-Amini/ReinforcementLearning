import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')

def featureExtraction(observation, mode=0):
    if not mode:  #  The features will be just observations!!!
        #  Turning obs into a 2d shape vector in numpy!!
        features = observation.copy()
        features = features[:, np.newaxis]
    elif mode == 1: #  The features are polynomials up to degree 2 of the observations!!
        length = observation.shape[0]
        print('obs', observation)
        print('len', length)
        features = np.zeros((int(length * (length + 1) / 2), 1))
        features[0:length, :] = observation[:, np.newaxis]
        counter = 0
        for i in range(length):
            for j in range(i + 1, length):
                features[length + counter] = observation[i] * observation[j]
                counter += 1
    #features = features[2:3, :]
    return features

def q(features, w):
    q_matrix = np.dot(w, features)
    return q_matrix

def actionsSelection(q_mat, actions_dict, eps=0.1):
    r = np.random.rand()
    if r>eps:
        return actions_dict[np.argmax(q_mat)], np.argmax(q_mat)
    else:
        a = np.random.randint(0, 2)
        return actions_dict[a], a

def updateWeights(w, action, td, features, alpha = 0.2):
    w[action:action+1, :] = w[action:action+1, :] + alpha * td * np.transpose(features)
    return w

actions_dict = {0: 1, 1: 0}
landa = 0.9
alpha = 0.001
feature_mode = 1
episode_no = 50
episode_lens = []
obs_no = 4
#w = np.array([[5], [-5]])
#w = np.zeros((2, 1))
w = np.zeros((2, 10))


for i_episode in range(episode_no):
    reward_history = []
    action_history = []
    end_counter = 0
    observation = env.reset()
    features = featureExtraction(observation, mode=feature_mode)
    for t in range(500):
        env.render()
        #action = env.action_space.sample()
        q_mat = q(features, w)
        action, action_index = actionsSelection(q_mat, actions_dict)
        observation, reward, done, info = env.step(action)
        action_history.append(action)

        if not done:
            target = reward + landa * np.max(q(featureExtraction(observation, mode=feature_mode), w))
            current_estimation = np.max(q_mat)
            temporal_difference = target - current_estimation
            w = updateWeights(w, action_index, temporal_difference, features, alpha)

            features = featureExtraction(observation, mode=feature_mode)

            print('angle: {}'.format(features[2, 0] * 180 / np.pi))
            print('features: {}'.format(features))
            print('q_mat')
            print(q_mat)
            print('action', action)
            print('weights: ')
            print(w)

        if done:
            end_counter += 1
            if end_counter > 50:
                episode_lens.append(t)
                print("Episode finished after {} timesteps".format(t+1))
                break

    episode_lens.append(t)

plt.figure()
plt.xticks([])
plt.plot([i for i in range(len(episode_lens))], episode_lens, 'ro--')
plt.show()
env.close()