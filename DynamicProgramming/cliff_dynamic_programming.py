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

def createA(gamma=1, probs=np.ones((48, 4)) * 0.25):
    A = np.eye(48)
    for s in range(37):
        for s_prime in neighbours(s):
            for r in [0, -1, -100]:
                for a in range(4):
                    A[s, s_prime] += -gamma * pi(a, s, probs) * p(s_prime, r, s, a)
    return A

def createB(probs=np.ones((48, 4)) * 0.25):
    B = np.zeros((48,1))
    for s in range(37):
        for s_prime in neighbours(s):
            for r in [-1, -100]:
                for a in range(4):
                    B[s] += pi(a, s, probs) * p(s_prime, r, s, a) * r
    B[37:47] = 0
    return B

def improvePolicy(v, probs, eps=0.5):
    for s in range(37):
        s_primes_a = [(nextState(s, a), a) for a in range(4)]
        s_primes_a = list(filter(lambda x: not isForbidden(x[0]), s_primes_a))
        vs = list(map(lambda x: (v[x[0]], x[1]), s_primes_a))
        v_max = max([v for (v, a) in vs])
        a_max = [a for (v, a) in vs if v == v_max]
        p_max = (1 - eps) / len(a_max)
        p_nonmax = eps / (4 - len(a_max))
        ps = list(map(lambda a: p_max if a in a_max else p_nonmax, range(4)))
        probs[s] = np.array(ps)
    return probs


def visualize(v):
    v_vis = np.zeros((4, 12), float)
    for i in range(4):
        for j in range(12):
            v_vis[i, j] = v[i * 12 + j, 0]
    plt.figure(figsize=(24, 8))
    sns.heatmap(v_vis, annot=True, fmt='.3f', linewidths=.5, cmap='YlGnBu')
    plt.title('Optimal Value Function')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('value_dp.png')
    plt.show()
    plt.close()

##  What if we set gamma to 0.5?! Why?!
probs = np.ones((48, 4)) * 0.25
gamma = 0.5
eps = 0.01
i = 0
while True:
    i -=- 1
    print(f'Iteration {i}...')
    A = createA(gamma=gamma, probs=probs)
    b = createB(probs)
    v = np.dot(np.linalg.inv(A), b)
    visualize(v)
    old_probs = probs.copy()
    probs = improvePolicy(v, probs, eps)
    if (probs == old_probs).all():
        print('No further improvement...')
        breakpoint()
        break




env = gym.make('CliffWalking-v0', render_mode='human')
observation, info = env.reset()

action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
v[37:47] = -1000

for _ in range(1000):
    s_prime = [nextState(observation, a) for a in range(4)]
    vs = [v[s, 0] for s in s_prime]
    action = np.argmax(vs)
    print(f'Observation: {observation}, Action: {action_dict[action]}')
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'Observation: {observation}')
    print(f'Reward: {reward}')
    print(f'Terminated: {terminated}')
    print(f'Truncated: {truncated}')
    print(f'Info: {info}')
    sleep(0.3)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
