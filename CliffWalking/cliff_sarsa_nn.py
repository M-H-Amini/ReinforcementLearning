#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein  Amini                              ##
##                        Title: Cliff - SARSA with Neural Network                     ##
##                                   Date: 2023/09/10                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 


import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import numpy as np
import logging as log
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from mh_qnetwork import generateQNetworks, updateModels, redundantTrain
from collections import Counter


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

def visualizePolicy(model):
    ##  Chars for up, right, down, left, circle are...
    chars = ['\u2191', '\u2192', '\u2193', '\u2190', '\u25CB']
    actions = np.full((4, 12), chars[4], dtype=str)
    for s in range(37):
        q = [model[i].predict(s2x(s), verbose=0)[0, 0] for i in range(4)]
        qmax = np.max(q)
        print('State: ', s, 'Q: ', q)
        a_max = [a for a in range(4) if q[a] == qmax]
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

def s2x(s):
    x = np.zeros((18,))
    ns = [isForbidden(t) for t in [s - 12, s + 1, s + 12, s - 1]]
    x[-4:] = ns
    r = s // 12
    c = s % 12
    x[r] = r
    x[c + 4] = c
    return x[np.newaxis, :]

def legalMoves(s):
    s_primes = [(a,nextState(s, a)) for a in range(4)]
    s_primes = [p for p in s_primes if p[1] != s]
    actions = [a for (a, _) in s_primes]
    return actions


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
        self.v = np.zeros((48,))
        self.gamma = gamma
        self.probs = np.ones((48, 4)) * 0.25
        self.eps = eps
        self.alpha = alpha
        self.landa = landa
        self.n_episodes = 0
        self.model_q = generateQNetworks()
        # self.model_q.compile(optimizer=SGD(learning_rate=alpha), loss='mse')
        self.model_q_a = generateQNetworks()
        # self.model_q_a = redundantTrain(self.model_q_a)
        # self.model_q = redundantTrain(self.model_q)
        self.cnt = Counter()
        self.X = [list() for _ in range(4)]
        self.X_prime = [list() for _ in range(4)]
        self.y_prime = [list() for _ in range(4)]
        self.y = [list() for _ in range(4)]
        self.done = [list() for _ in range(4)]
        print('Agent created...')

    def generateDataset(self):
        for i in range(len(self.hist_s)):
            self.X[self.hist_a[i]].append(s2x(self.hist_s[i])[0])
            g = float(self.hist_r[i])
            self.y_prime[self.hist_a[i]].append(g)
        
        for i in range(4):
            self.X[i] = np.array(self.X[i], dtype='float32')
            self.y_prime[i] = np.array(self.y_prime[i], dtype='float32')
        return self.X, self.y_prime


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
        self.cnt.update([(s, a)])
        if len(self.hist_s):
            x = s2x(self.hist_s[-1])
            a = self.hist_a[-1]
            r = self.hist_r[-1]
            x_prime = s2x(self.hist_s_prime[-1])
            self.X[a].append(x[0])
            self.X_prime[a].append(x_prime[0])
            self.y_prime[a].append(r)
            self.done[a].append(done)
            g_primes = [self.model_q_a[k].predict(x_prime, verbose=0)[0, 0] for k in range(4)]
            g_prime = np.max(g_primes)
            g = r + self.gamma * g_prime if not done else r
            self.y[a].append(g)
            # log.debug(f'Updating Q: s={s}, a={a}, r={r}, s_prime={s_prime}, a_prime={a_prime}')
            self.model_q = self.updateQ()   
        if done:
            self.n_episodes -=- 1 
            if self.n_episodes and not(self.n_episodes % 10):
                log.info(f'Episode: {self.n_episodes}, MeanG: {np.mean(Gs[-10:])}')
                visualizePolicy(self.model_q)
        if log.getLogger().isEnabledFor(log.DEBUG):
            input('Press any key to continue...')
        return self.model_q

    def clearBuffer(self):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
        self.z = np.zeros_like(self.g)
        self.X = [list() for _ in range(4)]
        self.X_prime = [list() for _ in range(4)]
        self.y_prime = [list() for _ in range(4)]
        self.y = [list() for _ in range(4)]
    
    def updateDataset(self, model):
        self.y = [list() for _ in range(4)]
        g_primes = [list() for _ in range(4)]
        for i in range(4):
            if len(self.X[i]):
                g_primes_ = np.concatenate([model[k].predict(np.array(self.X_prime[i]), verbose=0) for k in range(4)], axis=1)
                g_primes[i] = np.max(g_primes_, axis=1)
            
        for i in range(4):
            for j in range(len(self.X[i])):
                if self.done[i][j]:
                    g = self.y_prime[i][j]
                else:
                    g = self.y_prime[i][j] + self.gamma * g_primes[i][j]
                self.y[i].append(g)
        return self.X, self.y

    def updateQ(self, done=False, alpha=None):
        log.debug('Inside updateQ...')
        alpha = self.alpha if alpha is None else alpha
        X = [np.array(self.X[i], dtype='float32') for i in range(4)]
        y = [np.array(self.y[i], dtype='float32') for i in range(4)]
        self.model_q = updateModels(self.model_q, X, y)
        return self.model_q
   
    def act(self, s):
        x = s2x(s)
        actions = legalMoves(s)
        p = [(a, self.model_q_a[a].predict(x, verbose=0)[0]) for a in actions]
        qmax = np.max([q for (a, q) in p])
        a_max = [a for (a, q) in p if q == qmax]
        eps = np.random.rand()
        if eps > self.eps:
            return np.random.choice(a_max)
        else:
            a = np.random.choice(actions)
            return a
    
mha = MHAgent(gamma=0.99, eps=0.5, alpha=0.001, landa=0.9) 
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
        print(f'\nEpisode: {mha.n_episodes}, G: {G}, MeanG: {np.mean(Gs[-10:])}, Eps: {mha.eps}')
        observation, info = env.reset()  
        Gs.append(G)
        G = 0  
        if mha.n_episodes and not(mha.n_episodes % 3):
            mha.eps = mha.eps * 0.98
            for k in range(4):
                mha.model_q_a[k] = clone_model(mha.model_q[k])
                mha.updateDataset(mha.model_q_a)
                # mha.clearBuffer()
            print('Action model updated...')
    t -=- 1

