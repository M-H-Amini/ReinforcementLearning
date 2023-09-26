#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                     Title: Reinforce with Baseline on Funny Grid                    ##
##                                   Date: 2023/09/26                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

from mh_funnygrid_env import MHFunnyGrid
import matplotlib.pyplot as plt
import numpy as np

class MHReinforceAgent:
    def __init__(self, alpha=0.001, theta=0.5):
        self.theta = theta  ##  Probability of right action (a=1)
        self.alpha = alpha
        self.hist_a = []
        self.hist_r = []
        self.hist_done = []
        self.v = 0
        self.alpha_v = 0.01

    def updateBuffer(self, a, r, done):
        self.hist_a.append(a)
        self.hist_r.append(r)
        self.hist_done.append(done)

    def update(self):
        for i in range(len(self.hist_a)):
            g = sum(self.hist_r[i:])
            a = self.hist_a[i]
            self.v = self.v + self.alpha_v * (g - self.v)
            self.theta += self.alpha * (g - self.v) * self.dpi(a) / self.pi(a)
            self.theta = np.clip(self.theta, 0.01, 0.99)
        self.clearBuffer()

    def clearBuffer(self):
        self.hist_a = []
        self.hist_r = []
        self.hist_done = []

    def dpi(self, a):
        return 1 if a else -1

    def pi(self, a):
        return self.theta if a else 1 - self.theta

    def act(self):
        return np.random.choice([0, 1], p=[1-self.theta, self.theta])


mha = MHReinforceAgent(alpha=1/2**13, theta=0.3)
episodes = []
t = 0
env = MHFunnyGrid()
observation, info = env.reset()

for _ in range(100000):
    t -=- 1
    old_observation = observation
    action = mha.act()
    observation, reward, done, _, info = env.step(action)
    mha.updateBuffer(action, reward, done)
    if done:
        print('Done!')
        mha.update()
        episodes.append(t)
        print(f'Theta: {mha.theta}, V: {mha.v}')
        if len(episodes) > 100:
            break
        observation, info = env.reset()
        t = 0        

plt.figure(figsize=(12, 8))
plt.plot(range(len(episodes)), episodes, 'r-o')
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.ylim(0, 100)
plt.show()