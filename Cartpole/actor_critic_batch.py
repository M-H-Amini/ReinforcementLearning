from mh_networks import createCritic, createActor
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('models', exist_ok=True)

class MHActorCritic:
    def __init__(self, model_v_path=None, model_a_path=None, lr=0.0001):
        self.model_v = createCritic((4,)) if model_v_path is None else tf.keras.models.load_model(model_v_path)
        self.model_a = createActor((4,), 2) if model_a_path is None else tf.keras.models.load_model(model_a_path)
        self.opt_v = Adam(learning_rate=lr)
        self.opt_a = Adam(learning_rate=lr)
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
    
    def updateBuffer(self, s, a, r, s_prime, done):
        self.hist_s.append(s)
        self.hist_a.append(a)
        self.hist_r.append(r)
        self.hist_s_prime.append(s_prime)
        self.hist_done.append(done)
        # if len(self.hist_s) >= 32:
        s_list, a_list, r_list, s_prime_list, done_list = self.gatherExperience(n=31)
        s_list.append(s)
        a_list.append(a)
        r_list.append(r)
        s_prime_list.append(s_prime)
        done_list.append(done)
        self.update(s_list, a_list, r_list, s_prime_list, done_list)

    def clearBuffer(self):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []

    def s2x(self, s):
        return np.expand_dims(s, 0)

    def update(self, s_list, a_list, r_list, s_prime_list, done_list):
        bs = len(s_list)
        x = np.concatenate([self.s2x(s) for s in s_list], axis=0)
        x_prime = np.concatenate([self.s2x(s_prime) for s_prime in s_prime_list], axis=0)
        v_s = self.model_v(x).numpy()
        # print('\r', end='')
        # print(f'v_s: {v_s.numpy():4.2f}', end='')
        v_s_prime = self.model_v(x_prime).numpy()
        diff_list = np.array([r - v if done else r + v_prime - v for r, done, v, v_prime in zip(r_list, done_list, v_s, v_s_prime)])
        with tf.GradientTape(persistent=True) as tape:
            v_s = - diff_list * self.model_v(x) / bs
            p_list = []
            for a, s in zip(a_list, s_list):
                if a:
                    p = self.model_a(self.s2x(s))
                else:
                    p = 1 - self.model_a(self.s2x(s))
                p_list.append(p)
            

            log_p = tf.math.log(p_list)[..., 0]
            # print('log_p: ', log_p.shape, end='')
            # print(f'\ta: {a} r: {r:.2f}, p: {p.numpy()[0,0]:.2f}, log_p: {log_p.numpy()[0,0]:.2f}, diff: {diff:4.2f}', end='')
            log_p = -diff_list * log_p / bs
            # print('log_p: ', log_p.shape, end='')

        grad_v = tape.gradient(v_s, self.model_v.trainable_variables)
        self.opt_v.apply_gradients(zip(grad_v, self.model_v.trainable_variables))
        grad_a = tape.gradient(log_p, self.model_a.trainable_variables)
        self.opt_a.apply_gradients(zip(grad_a, self.model_a.trainable_variables))


    def gatherExperience(self, n=32):
        indices = np.random.choice(range(len(self.hist_s)), size=(n,))
        s_list = [self.hist_s[i] for i in indices]
        a_list = [self.hist_a[i] for i in indices]
        r_list = [self.hist_r[i] for i in indices]
        s_prime_list = [self.hist_s_prime[i] for i in indices]
        done_list = [self.hist_done[i] for i in indices]
        return s_list, a_list, r_list, s_prime_list, done_list
        

    def act(self, observation):
        if len(observation.shape) == 1:
            observation = observation[np.newaxis, ...]
        phi = self.model_a.predict(observation, verbose=0)[0, 0]
        return np.random.choice([0, 1], p=[1 - phi, phi])

def generateReward(observation, method='theta'):
    if method and 'theta' in method.lower():
        x, x_dot, theta, theta_dot = observation
        r = (1 - abs(theta) * 5) * 10
    else:
        r = 1
    return r


model_v_path = 'models/model_v'
model_a_path = 'models/model_a'
mha = MHActorCritic(lr=0.00001)
# mha = MHActorCritic(model_v_path=model_v_path, model_a_path=model_a_path, lr=0.00001)
# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')

observation, info = env.reset()
t = 0
n_episode = 0
episodes = []
best_mean = -100

while n_episode < 2000:
    t -=- 1
    old_observation = observation
    action = mha.act(observation)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    reward = generateReward(observation, None)
    mha.updateBuffer(old_observation, action, reward, observation, terminated or truncated)
    if terminated or truncated:
        n_episode -=- 1
        episodes.append(t)
        if np.mean(episodes[-10:]) > best_mean:
            best_mean = np.mean(episodes[-10:])
            mha.model_v.save(model_v_path)
            mha.model_a.save(model_a_path)
            print(f'\nSaved models. Best mean: {best_mean:.2f}')
        print(f'\nDone! Episode {n_episode} length: {t}, mean: {np.mean(episodes[-10:]):.2f}, best mean: {best_mean:.2f}')
        observation, info = env.reset()
        t = 0

env.close()

plt.figure(figsize=(12, 8))
plt.plot(range(len(episodes)), episodes, 'r-o')
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.savefig('CartPole-ActorCritic.png')
plt.show()

