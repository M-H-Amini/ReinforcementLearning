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

def createA(gamma=1):
    A = np.eye(38)
    for s in range(37):
        for s_prime in neighbours(s):
            s_prime_ = 37 if s_prime == 47 else s_prime
            for r in [-1, -100]:
                for a in range(4):
                    A[s, s_prime_] += -gamma * pi(a, s) * p(s_prime, r, s, a)
    return A

def createB():
    B = np.zeros((38,1))
    for s in range(37):
        for s_prime in neighbours(s):
            s_prime_ = 37 if s_prime == 47 else s_prime
            for r in [-1, -100]:
                for a in range(4):
                    B[s] += pi(a, s) * p(s_prime, r, s, a) * r
    B[37] = 0
    return B


def visualize(v):
    v_vis = np.zeros((4, 12), float)
    for i in range(3):
        for j in range(12):
            v_vis[i, j] = v[i * 12 + j, 0]
    v_vis[3, 0] = v[36, 0]
    plt.figure(figsize=(20, 8))
    sns.heatmap(v_vis, annot=True, fmt='.1f', linewidths=.5, cmap='YlGnBu')
    plt.title('Optimal Value Function')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()
    plt.close()

##  What if we set gamma to 0.5?! Why?!
A = createA(gamma=0.999)
b = createB()
v = np.dot(np.linalg.inv(A), b)
visualize(v)


env = gym.make('CliffWalking-v0', render_mode='human')
observation, info = env.reset()

action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

for _ in range(1000):
    s_prime = [nextState(observation, a)[0] for a in range(4)]
    if 47 in s_prime:
        s_prime = [s if s != 47 else 37 for s in s_prime]
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
