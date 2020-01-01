import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')

def featureExtraction(observation):
    #  Turning obs into a 2d shape vector in numpy!!
    features = observation.copy()
    features = features[:, np.newaxis]

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
episode_no = 50
episode_lens = []
#w = np.array([[5], [-5]])
#w = np.zeros((2, 1))
w = np.zeros((2, 4))


for i_episode in range(episode_no):
    reward_history = []
    action_history = []
    end_counter = 0
    observation = env.reset()
    features = featureExtraction(observation)
    for t in range(500):
        env.render()
        #action = env.action_space.sample()
        q_mat = q(features, w)
        action, action_index = actionsSelection(q_mat, actions_dict)
        observation, reward, done, info = env.step(action)
        action_history.append(action)

        if not done:
            target = reward + landa * np.max(q(featureExtraction(observation), w))
            current_estimation = np.max(q_mat)
            temporal_difference = target - current_estimation
            w = updateWeights(w, action_index, temporal_difference, features, alpha)

            features = featureExtraction(observation)

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