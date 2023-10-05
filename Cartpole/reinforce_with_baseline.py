#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                            Title: Reinforce with Baseline                           ##
##                                   Date: 2023/10/05                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 




import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

def generateActorNetwork(input_shape=(4,), actions=2):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    outputs = Dense(actions, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def generateCriticNetwork(input_shape=(4,)):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class MHReinforce:
    def __init__(self, no_of_actions=2, state_dim=4):
        self.n_action = no_of_actions
        self.state_dim = state_dim
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []
        self.model_a = generateActorNetwork(input_shape=(state_dim,), actions=no_of_actions)
        self.opt_a = Adam(0.01)
        self.model_v = generateCriticNetwork(input_shape=(state_dim,))
        self.model_v.compile(loss='mse', optimizer=Adam(0.01))

    def updateBuffer(self, s, a, r, s_prime, done):
        self.hist_s.append(s)
        self.hist_a.append(a)
        self.hist_r.append(r)
        self.hist_s_prime.append(s_prime)
        self.hist_done.append(done)

    def clearBuffer(self):
        self.hist_s = []
        self.hist_a = []
        self.hist_r = []
        self.hist_s_prime = []
        self.hist_done = []

    def updateCritic(self, verbose=0):
        N = len(self.hist_s)
        G = []
        for i in range(N):
            G.append(sum([r for r in self.hist_r[i:]]))
        X = np.array([s for s in self.hist_s])
        G = np.array(G)
        self.model_v.fit(X, G, epochs=20, verbose=verbose)
        loss = tf.losses.mean_squared_error(G, self.model_v.predict(X, verbose=0)[:, 0])
        return loss.numpy()

    def updateActor(self):
        N = len(self.hist_s)
        G = []
        for i in range(N):
            G.append(sum([r for r in self.hist_r[i:]]))
        X = np.array([s for s in self.hist_s])
        G = np.array(G)
        diff = G - self.model_v.predict(X, verbose=0)[:, 0]
        log_probs = []
        losses = []
        with tf.GradientTape() as tape:
            for a, x, d in zip(self.hist_a, self.hist_s, diff):
                log_probs.append(tf.math.log(self.model_a(x[np.newaxis, ...])[0, a]))
                losses.append(-d * log_probs[-1])
            loss = tf.reduce_mean(losses)
        grad_a = tape.gradient(loss, self.model_a.trainable_variables)
        self.opt_a.apply_gradients(zip(grad_a, self.model_a.trainable_variables))
        return loss.numpy()

    def updateEpisode(self):
        loss_v = self.updateCritic(verbose=0)
        loss_a = self.updateActor()
        print(f'Loss_v: {loss_v:4.2f}, Loss_a: {loss_a:4.2f}')
        self.clearBuffer()

    def act(self, s):
        p = self.model_a(np.expand_dims(s, 0)).numpy()[0]
        a = np.random.choice(len(p), p=p)
        print('\r', end='')
        print(f'p = {p[a]:4.2f}', end='')
        return a


render_mode = None
# render_mode = 'human'
env = gym.make('CartPole-v1', render_mode=render_mode)
# env = gym.make('LunarLander-v2', render_mode=render_mode)
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
mha = MHReinforce(no_of_actions=n_actions, state_dim=n_states)
max_episode_steps = 500

observation, info = env.reset()

t = 0
g = 0
n_episode = 0
while True:
    old_observation = observation
    action = mha.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    mha.updateBuffer(old_observation, action, reward, observation, done:= (terminated or truncated))
    t -=- 1
    g += reward
    if done:
        n_episode -=- 1
        mha.updateEpisode()
        print(f'Episode {n_episode} finished after {t} timesteps with total reward {g}')
        observation, info = env.reset()
        t = 0
        g = 0


env.close()