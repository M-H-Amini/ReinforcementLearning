import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def pi(a, theta):
    return theta if a else 1 - theta

def nextState(s, a):
    a_dict = {0: -1, 1:1}
    s_prime = s
    if s != 1:
        s_prime = s + a_dict[a]
    else:
        s_prime = s - a_dict[a]
    s_prime = np.clip(s_prime, 0, 3)
    return s_prime

def p(s_prime, r, s, a):
    s_next = nextState(s, a)
    return 1 if (s_prime, r) == (s_next, -1) else 0

def createA(theta):
    A = np.eye(4)
    for s in range(3):
        for s_prime in range(4):
            for a in range(2):
                A[s, s_prime] -= pi(a, theta) * p(s_prime, -1, s, a)
    return A

def createB(theta):
    b = np.zeros((4, 1))
    for s in range(3):
        for s_prime in range(4):
            for a in range(2):
                b[s] += pi(a, theta) * p(s_prime, -1, s, a) * -1
    b[3] = 0
    return b 

def findV(theta):
    A = createA(theta)
    b = createB(theta)
    v = np.dot(np.linalg.inv(A), b)
    return v[0]

theta = np.linspace(0.01, 0.99, 1000)
vs = [findV(t) for t in tqdm(theta)]
print(f'The best theta is: {theta[np.argmax(vs)]} with v: {np.max(vs)}')
plt.figure(figsize=(12, 8))
plt.plot(theta, vs, 'r-')
plt.xlabel('Theta')
plt.ylabel('V')
plt.title('V vs Theta for FunnyGrid')
plt.savefig('FunnyGrid.png')
plt.show()
