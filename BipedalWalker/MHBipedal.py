import gym
import numpy as np 

env = gym.make('BipedalWalker-v3')
a0, a1, b0, b1 = 0, 0, 0, 0

for i_episode in range(20):
    observation = env.reset()
    for t in range(500):
        env.render()
        # print(observation)
        a0 = np.sin(2*np.pi*t/10)
        a1 = -a0
        b0 = -1
        b1 = 1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(np.array([a0, a1, b0, b1]))
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()