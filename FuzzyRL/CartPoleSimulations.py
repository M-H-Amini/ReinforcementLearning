import gym
import numpy as np
from matplotlib import pyplot as plt
import MHFuzzy as mh
from MHFuzzy import *

env = gym.make('CartPole-v0')

def featureExtraction(observation, mode=0):
    if mode==1:  #  The features will be just observations!!!
        #  Turning obs into a 2d shape vector in numpy!!
        features = observation.copy()
        features = features[:, np.newaxis]
    elif mode == -1:
        #  The features will be just observations with a bias!!!
        length = observation.shape[0]
        features = np.zeros((length + 1, 1))
        features[0:length, :] = observation[:, np.newaxis]
        features[length, :] = 1
    elif mode == -2:
        #  The features are polynomials up to degree 2 of the observations!!
        #  with no x^2 term there! only cross-terms!
        length = observation.shape[0]
        features = np.zeros((int(length * (length + 1) / 2) + 1, 1))
        features[0:length, :] = observation[:, np.newaxis]
        counter = 0
        for i in range(length):
            for j in range(i + 1, length):
                features[length + counter] = observation[i] * observation[j]
                counter += 1
        features[-1:, :] = 1

    elif mode == 2:
        #  The features are polynomials up to degree 2 of the observations!!
        #  with no x^2 term there! only cross-terms!
        length = observation.shape[0]
        features = np.zeros((int(length * (length + 1) / 2), 1))
        features[0:length, :] = observation[:, np.newaxis]
        counter = 0
        for i in range(length):
            for j in range(i + 1, length):
                features[length + counter] = observation[i] * observation[j]
                counter += 1
    elif mode == -3:
        #  The features are polynomials up to degree 2 of the observations!!
        #  with x^2 term there! There will be a bias term too!
        length = observation.shape[0]
        features = np.zeros((int(length * (length + 3) / 2) + 1, 1))
        features[0:length, :] = observation[:, np.newaxis]
        counter = 0
        for i in range(length):
            for j in range(i, length):
                features[length + counter] = observation[i] * observation[j]
                counter += 1
        features[-1:, :] = 1
    elif mode == 3:
        #  The features are polynomials up to degree 2 of the observations!!
        #  with x^2 terms there!
        length = observation.shape[0]
        features = np.zeros((int(length * (length + 3) / 2), 1))
        features[0:length, :] = observation[:, np.newaxis]
        counter = 0
        for i in range(length):
            for j in range(i, length):
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
        #print('random!', a)
        return actions_dict[a], a

def updateWeights(w, action, td, features, alpha = 1):
    #print('&'*50, 'updateWeights Func')
    #print('features: ')
    #print(np.transpose(features))
    #print('&'*50)
    w[action:action+1, :] = w[action:action+1, :] + alpha * td * np.transpose(features)
    return w

def temporalDifference(action_history, reward_history, features_history, gamma, w):
    #print('x'*50, 'TD Func')
    target = reward_history[-2] + gamma * np.max(q(features_history[-1], w))
    current_estimation = np.max(q(features_history[-2], w))
    #print('target: {} \t current_est: {}'.format(target, current_estimation))
    td = target - current_estimation
    #print('td: {}'.format(td))
    return td

def temporalDifferenceN(n, action_history, reward_history, features_history, gamma, w):
    target = 0
    for i in range(n):
        target += reward_history[-i-2] * (gamma ** (n - i - 1))
    target += (gamma**n) * np.max(q(features_history[-1], w))
    current_estimation = np.max(q(features_history[-n -1], w))
    td = target - current_estimation
    return td

def distance(x):
    return np.sqrt(x[0]**2 + x[2]**2)

#  To be done later...
def lambdaReturn():
    pass

def runCart(times, no_of_episodes=50, eps=0.1 , output_file='Cartpole', fuzzy=False):
    data = []
    actions_dict = {0: 1, 1: 0}
    gamma = 0.8
    alpha = 0.03
    feature_mode = -2
    episode_no = no_of_episodes
    eps = eps
    tdN = 4

    #  Fuzzy part!
    small = MHMember(0, 0, 0.1, 0.15)
    medium = MHMember(0.1, 0.2, 0.3, 0.4)
    big = MHMember(0.35, 0.5, 0.5, 0.5)
    close = MHMember(0, 0, 0, 0.2)
    far = MHMember(0.1, 1, 100, 100)
    rule1 = MHRule([close], [small])
    rule2 = MHRule([far], [big])
    fis = MHFIS([rule1, rule2])
    #print(fis.output([20])
    show_details = False
    #w = np.array([[5], [-5]])
    #w = np.zeros((2, 1))
    #w = np.array([[0, 0, 0.1, 0, 5], [0, 0, -0.1, 0, 5]]) #  For mode=-1 featureExtraction
    #w = np.zeros((2, 14))  #  For mode=2 featureExtraction
    #w = np.zeros((2, 4))  #  For mode=0 featureExtraction
    #w = np.zeros((2, 10))  #  For mode=1 featureExtraction
    #w = np.zeros((2, 14))  #  For mode=2 featureExtraction

    for time in range(times):
        w = np.zeros((2, 11))  # For mode=-2 featureExtraction
        # w = np.zeros((2, 15))  #  For mode=-3 featureExtraction
        w[0, 2] = 0.1
        w[1, 2] = -0.1
        w[0, -1] = 5
        w[1, -1] = 5
        print('*'*24, 'Time {}'.format(time + 1), '*'*24)
        episode_lens = []
        for i_episode in range(episode_no):
            reward_history = []
            action_history = []
            features_history=[]
            eps_history = []
            end_counter = 0

            observation = env.reset()
            features = featureExtraction(observation, mode=feature_mode)
            features_history.append(features)

            if show_details:
                print('-'*20,'Episode {}'.format(i_episode), '-'*20)

            for t in range(500):
                #alpha *= np.exp(-t/500)
                #env.render()
                #action = env.action_space.sample()
                if fuzzy:
                    eps = fis.output([distance(features)])
                    eps_history.append(eps)
                    #print(20*'_', 'distance={}'.format(distance(features)), 'eps={}'.format(eps), 20*'_')

                q_mat = q(features, w)
                action, action_index = actionsSelection(q_mat, actions_dict, eps=eps)
                observation, reward, done, info = env.step(action)


                if not done:
                    angle = observation[2] * 180 / np.pi
                    #target = reward + gamma * np.max(q(featureExtraction(observation, mode=feature_mode), w))
                    #current_estimation = np.max(q_mat)
                    #temporal_difference = target - current_estimation
                    action_history.append(action_index)
                    reward_history.append(reward)
                    if t > tdN - 1:
                        #temporal_difference = temporalDifference(action_history, reward_history, features_history, gamma, w)
                        temporal_difference = temporalDifferenceN(tdN, action_history, reward_history, features_history, gamma, w)
                        #print('in While', '-'*50)
                        #print(features_history)
                        w = updateWeights(w, action_history[-tdN -1], temporal_difference, features_history[-tdN -1], alpha=alpha)

                    features_history.append(features)
                    features = featureExtraction(observation, mode=feature_mode)

                    if show_details:
                        print('episode: {} \t t: {}'.format(i_episode, t))
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
                        #print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                        break

            episode_lens.append(t)
        data.append(np.array(episode_lens))

    data_array = np.array(data)
    np.save(output_file, data_array)
    np.save('eps', np.array(eps_history))
    print('Done!!!')

output_file_common_name = 'FuzzyCartpole'
runCart(100, output_file=output_file_common_name, fuzzy=True)
'''
runCart(100, eps=0.1, output_file='Cartpole-eps0.1')
runCart(100, eps=0.15, output_file='Cartpole-eps0.15')
runCart(100, eps=0.2, output_file='Cartpole-eps0.2')
runCart(100, eps=0.25, output_file='Cartpole-eps0.25')
runCart(100, eps=0.3, output_file='Cartpole-eps0.3')
runCart(100, eps=0.35, output_file='Cartpole-eps0.35')
runCart(100, eps=0.4, output_file='Cartpole-eps0.4')
runCart(100, eps=0.45, output_file='Cartpole-eps0.45')
runCart(100, eps=0.5, output_file='Cartpole-eps0.5')
'''

env.close()