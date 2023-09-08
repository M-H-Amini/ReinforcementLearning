#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein  Amini                              ##
##                               Title: Cliff - Q Learning                             ##
##                                   Date: 2023/08/21                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 


import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import numpy as np
from collections import Counter
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_theme()
plt.rcParams['font.family'] = 'DejaVu Sans'

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
    n = list(set([nextState(s, a) for a in range(4)]))
    n = list(filter(lambda x: (isValid(*state2Coord(x))), n))
    return n

def nextState(s, a):
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

def nextReward(s, a):
    s_prime = nextState(s, a)
    return -100 if isForbidden(s_prime) else (0 if s_prime == 47 else -1)

def p(s_prime, r, s, a):
    s_next = nextState(s, a)
    r_next = nextReward(s, a)
    return 1 if (s_prime, r) == (s_next, r_next) else 0


def pi(a, s, probs=np.ones((48, 4)) * 0.25):    
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
    plt.close('all')

def visualizePolicy(q):
    ##  Chars for up, right, down, left, circle are...
    chars = ['\u2191', '\u2192', '\u2193', '\u2190', '\u25CB']
    actions = np.full((4, 12), chars[4], dtype=str)
    for s in range(37):
        qmax = np.max(q[s])
        a_max = [a for a in range(4) if q[s, a] == qmax]
        if len(a_max) == 1:
            actions[state2Coord(s)] = chars[a_max[0]]
        elif len(a_max) == 2:
            if 0 in a_max and 1 in a_max:
                actions[state2Coord(s)] = '\u2197'
            elif 1 in a_max and 2 in a_max:
                actions[state2Coord(s)] = '\u2198'
            elif 2 in a_max and 3 in a_max:
                actions[state2Coord(s)] = '\u2199'
            elif 3 in a_max and 0 in a_max:
                actions[state2Coord(s)] = '\u2196' #  chars[3] + chars[0]
            elif 0 in a_max and 2 in a_max:
                actions[state2Coord(s)] = chars[4] #  chars['lr']
            elif 1 in a_max and 3 in a_max:
                actions[state2Coord(s)] = chars[4] #  chars['ud']
        else:
            actions[state2Coord(s)] = chars[4]
    
    plt.figure(figsize=(24, 8))
    actions_arr = [[100 for _ in range(12)] for _ in range(4)]
    sns.heatmap(actions_arr, annot=actions, fmt='s', linewidths=.5, annot_kws={'fontsize':28})
    plt.title('Optimal Policy')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('policy_q_learning.png')
    plt.show()
    plt.close('all')


def isTerminated(s, r):
    return True if r == 0 or r == -100 else False

class MHAgent:
    def __init__(self, gamma=1, eps=0.5, alpha=0.5, landa=0):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
        self.g = np.zeros((48,4))
        self.q = np.zeros_like(self.g)
        self.z = np.zeros_like(self.g)
        self.N = Counter()
        self.v = np.zeros((48,))
        self.gamma = gamma
        self.probs = np.ones((48, 4)) * 0.25
        self.eps = eps
        self.alpha = alpha
        self.landa = landa
        self.n_episodes = 0
        print('Agent created...')

    def updateBuffer(self, s, a, s_prime, r, done, alpha=None):
        if s_prime == 47:
            r = 0
        self.hist_s.append(s)
        self.hist_a.append(a)
        if s_prime == 36 and r == -100:
            s_prime = 37
        self.hist_s_prime.append(s_prime)
        self.hist_r.append(r)
        self.hist_done.append(done)
        log.debug(f'Buffer updated: s={s}, a={a}, s_prime={s_prime}, r={r}, done={done}')
        log.debug(f'Buffer hist_s: {self.hist_s}')
        log.debug(f'Buffer hist_a: {self.hist_a}')
        log.debug(f'Buffer hist_s_prime: {self.hist_s_prime}')
        log.debug(f'Buffer hist_r: {self.hist_r}')
        log.debug(f'Buffer hist_done: {self.hist_done}')
        if len(self.hist_s) > 1:
            s, a, r, s_prime = self.hist_s[-2], self.hist_a[-2], self.hist_r[-2], self.hist_s_prime[-2]
            a_prime = self.hist_a[-1]
            log.debug(f'Updating Q: s={s}, a={a}, r={r}, s_prime={s_prime}, a_prime={a_prime}')
            self.v, self.q = self.updateQ(s, a, r, s_prime, a_prime)
        if done:
            s, a, r, s_prime = self.hist_s[-1], self.hist_a[-1], self.hist_r[-1], self.hist_s_prime[-1]
            a_prime = self.act(s_prime)
            log.debug(f'Updating Q: s={s}, a={a}, r={r}, s_prime={s_prime}, a_prime={a_prime}')
            self.v, self.q = self.updateQ(s, a, r, s_prime, a_prime)
            self.clearBuffer()
            self.n_episodes -=- 1
            # if not(self.n_episodes % 10):
        if log.getLogger().isEnabledFor(log.DEBUG):
            input('Press any key to continue...')
        self.improvePolicy()
        return self.v, self.q

    def clearBuffer(self):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
        self.z = np.zeros_like(self.g)
            

    def updateQ(self, s, a, r, s_prime, a_prime, alpha=None):
        log.debug('Inside updateQ...')
        alpha = self.alpha if alpha is None else alpha
        for ss in range(48):
            for aa in range(4):
                if s == ss and a == aa:
                    self.z[s, a] = self.gamma * self.landa * self.z[s, a] + 1
                else:
                    self.z[ss, aa] = self.gamma * self.landa * self.z[ss, aa]
        log.debug(f's={s}, a={a}, s_prime={s_prime}, a_prime={a_prime}')
        log.debug(f'alpha={alpha}, gamma={self.gamma}, landa={self.landa}')
        log.debug(f'z=\n{self.z}')
        g = r + self.gamma * self.q[s_prime, a_prime]
        log.debug(f'g={g}')
        diff = g - self.q[s, a]
        log.debug(f'diff={diff}')
        for ss in range(48):
            for aa in range(4):
                self.q[ss, aa] = self.q[ss, aa] + alpha * self.z[ss, aa] * diff
            self.v[ss] = sum([self.probs[ss, aa] * self.q[ss, aa] for aa in range(4)])
        log.debug(f'q=\n{self.q}')
        return self.v, self.q
    
    def improvePolicy(self):
        qmax = np.max(self.q, axis=1)
        a_max = ~(self.q - np.expand_dims(qmax, 1)).astype(bool)
        n_a_max = np.sum(a_max, axis=1)
        for s in range(37):
            if n_a_max[s] != 4:
                p_max = (1 - self.eps) / n_a_max[s]
                p_nonmax = self.eps / (4 - n_a_max[s])
            else:
                p_max = 0.25
                p_nonmax = 0
            ps = list(map(lambda a: p_max if a_max[s, a] else p_nonmax, range(4)))
            self.probs[s] = np.array(ps)
        return self.probs
    
    def act(self, s):
        return np.random.choice(4, p=self.probs[s])
    


mha = MHAgent(gamma=0.99, eps=0.1, alpha=0.8, landa=0.9) 
# env = gym.make('CliffWalking-v0', render_mode='human')
env = gym.make('CliffWalking-v0')
observation, info = env.reset()
action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

t = 0
probs = np.ones((48, 4)) * 0.25
Gs = []
G = 0
while True:
    action = mha.act(observation)
    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    reward = 0 if observation == 47 else reward
    G += reward
    mha.updateBuffer(old_observation, action, observation, reward, done:=isTerminated(observation, reward))
    if terminated or truncated or done:
        observation, info = env.reset()  
        Gs.append(G)
        G = 0  
    if t and not t % 1000:
        print(f'Timestep: {t}, MeanG: {np.mean(Gs[-10:])}')
        if not t % 1000:
            visualizePolicy(mha.q)
            visualize(mha.v)
            command = input('What to do?')
            if command.lower() in ['exit', 'q']:
                break
    t += 1


plt.figure()
plt.plot(range(len(Gs)), Gs, 'r-o')
plt.savefig('cliff_mc_returns.png')
plt.show()
breakpoint()
env.close()
