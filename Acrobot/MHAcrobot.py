import gym
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import MHModel as mhm

env = gym.make('Acrobot-v1')

model = mhm.buildModel()

model_train = Sequential([
    model,
    Dense(1)
])

model_train.compile(optimizer='adam', loss='mse')

hist_obs = []
hist_act = []
hist_rew = []

for i_episode in range(20):
    hist_obs = []
    hist_act = []
    hist_rew = []
    observation = env.reset()
    hist_obs.append(observation)
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        hist_act.append(action)
        observation, reward, done, info = env.step(action)
        hist_rew.append(reward)
        hist_obs.append(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()