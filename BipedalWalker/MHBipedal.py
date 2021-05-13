import gym

env = gym.make('BipedalWalker-v3')
env.reset()

for _ in range(1000):
    env.render()
    print(env.action_space.sample())
    env.step(env.action_space.sample()) # take a random action

env.close()