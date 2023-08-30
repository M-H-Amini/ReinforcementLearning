#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein  Amini                              ##
##                              Title: Cliff - DP Iterative                            ##
##                                   Date: 2023/08/21                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Dynamic Programming using Linear Equations...


import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import numpy as np
from time import sleep
sns.set_theme()

def isForbidden(s):
    return s in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

def state2Coord(s):
    return s // 12, s % 12

def coord2State(r, c):
    return r * 12 + c

def isValid(r, c):
    if r < 0 or r > 3:
        return False 
    if c < 0 or c > 11:
        return False 
    return True

def neighbours(s):
    r, c = state2Coord(s)
    n = [(r, c), (r+1, c), (r-1, c), (r, c+1), (r, c-1)]
    n = list(filter(lambda x: (isValid(*x) and not isForbidden(coord2State(*x))), n))
    ##  Add 36 if we are beside forbidden states...
    n += [(3, 0)] if r == 2 and 1 <= c <= 10 else []
    return [coord2State(*x) for x in n]

def _nextState(s, a):
    r, c = state2Coord(s)
    r_prime, c_prime = r, c
    if a == 0:  ##  up...
        r_prime = r - 1
    elif a == 1:  ##  right...
        c_prime = c + 1
    elif a == 2:  ##  down...
        r_prime = r + 1
    elif a == 3:  ## left...
        c_prime = c - 1
    if not isValid(r_prime, c_prime):
        r_prime, c_prime = r, c
    return coord2State(r_prime, c_prime)

def nextState(s, a):
    s_prime_logical = _nextState(s, a)
    s_prime = 36 if isForbidden(s_prime_logical) else s_prime_logical
    return s_prime, s_prime_logical

def nextReward(s, a):
    _, s_prime = nextState(s, a)
    return -100 if isForbidden(s_prime) else (0 if s_prime == 47 else -1)

def p(s_prime, r, s, a):
    s_next, _ = nextState(s, a)
    r_next = nextReward(s, a)
    return 1 if (s_prime, r) == (s_next, r_next) else 0


def pi(a, s):
    probs = np.ones((38, 4)) * 0.25
    return probs[s][a]

def visualize(v):
    v_vis = np.zeros((4, 12), float)
    for i in range(4):
        for j in range(12):
            v_vis[i, j] = v[i * 12 + j]
    plt.figure(figsize=(24, 8))
    sns.heatmap(v_vis, annot=True, fmt='.3f', linewidths=.5, cmap='YlGnBu')
    plt.title('Optimal Value Function')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('value_mc.png')
    plt.show()
    plt.close()

def isTerminated(s, r):
    return True if r == 0 or r == -100 else False

class MHAgent:
    def __init__(self, gamma=1):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
        self.g = [0 for _ in range(48)]
        self.v = [0 for _ in range(48)]
        self.N = [0 for _ in range(48)]
        self.gamma = gamma
        print('Agent created...')

    def updateBuffer(self, s, a, s_prime, r, done):
        if s_prime == 47:
            r = 0
        self.hist_s.append(s)
        self.hist_a.append(a)
        self.hist_s_prime.append(s_prime)
        self.hist_r.append(r)
        self.hist_done.append(done)
        ##  Update value if done...
        if done:
            for t, s in enumerate(self.hist_s):
                self.g[s] += sum([self.hist_r[n] * self.gamma ** (n-t) for n in range(t, len(self.hist_s))])
                self.N[s] -=- 1
            self.clearBuffer()

    def clearBuffer(self):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
            

    def updateV(self):
        for s in range(len(self.v)):
            if self.N[s]:
                self.v[s] = self.g[s] / self.N[s]
        return self.v


mha = MHAgent(gamma=0.99)
    
# env = gym.make('CliffWalking-v0', render_mode='human')
env = gym.make('CliffWalking-v0')
observation, info = env.reset()

action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

t = 0
while True:
    action = np.random.randint(4)
    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    reward = 0 if observation == 47 else reward
    mha.updateBuffer(old_observation, action, observation, reward, done:=isTerminated(observation, reward))
    if terminated or truncated or done:
        observation, info = env.reset()    
    if t and not t % 1000:
        v = mha.updateV()
        print(f'Timestep: {t}, N[35] = {mha.N[35]}')
        if mha.N[35] >= 500:
            break
    t += 1


v = mha.updateV()
visualize(v)

breakpoint()

env.close()
