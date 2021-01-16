import numpy as np 
import gym
from time import sleep 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def saveModel():
    model.save('model')

def loadModel():
    model = keras.models.load_model('model')
    return model

# model = loadModel()

model = Sequential([
    Dense(4, 'tanh', input_shape=(4,)),
    Dense(2)
])

model_train = Sequential([
    model,
    Dense(1)
])

model_train.compile(optimizer=Adam(1.), loss='mse')

def processModel(model, i):
    w = np.zeros((2, 1))
    w[i, 0] = 1
    model.layers[-1].set_weights([w, np.array([0])])
    model.layers[-1].trainable = False


# print(model_train.layers[-1].get_weights())
# print(model_train.layers[-1].trainable)
# processModel(model_train, 1)
# print(model_train.layers[-1].get_weights())
# print(model_train.layers[-1].trainable)
# print(model(np.array([[4, 2, 1., 0.5]])))
# print(model_train(np.array([[4, 2, 1., 0.5]])))


def findEpisodeStats(hist_obs, hist_act, hist_rew):
    print('Before Training: ')
    X_batch_l, y_batch_l = [], []
    X_batch_r, y_batch_r = [], []
    for i in range(len(hist_act)):
        ret = sum(hist_rew[i:])
        # if len(hist_act) >= 199 and i > 100:
        #     ret = 200
        if hist_act[i]==0:
            X_batch_l.append(hist_obs[i])
            y_batch_l.append(ret)
            print(f'Actual: {ret}\t Predicted: {model.predict(np.array([hist_obs[i]]))}')
        else:
            X_batch_r.append(hist_obs[i])
            y_batch_r.append(ret)
            print(f'Actual: {ret}\t Predicted: {model.predict(np.array([hist_obs[i]]))}')

    X_batch_l = np.array(X_batch_l)
    X_batch_r = np.array(X_batch_r)
    y_batch_l = np.array(y_batch_l)
    y_batch_r = np.array(y_batch_r)
    print(len(X_batch_l), len(y_batch_l), len(X_batch_r), len(y_batch_r))
    print(X_batch_l.shape, X_batch_r.shape)
    for i in range(2):
        if len(X_batch_l):
            processModel(model_train, 0)
            model_train.train_on_batch(X_batch_l, y_batch_l)
        if len(X_batch_r):
            processModel(model_train, 1)
            model_train.train_on_batch(X_batch_r, y_batch_r)
            

    print('After Training: ')

    for i in range(len(hist_act)):
        ret = sum(hist_rew[i:])
        if hist_act[i]==0:
            print(f'Actual: {ret}\t Predicted: {model.predict(np.array([hist_obs[i]]))}')
        else:
            print(f'Actual: {ret}\t Predicted: {model.predict(np.array([hist_obs[i]]))}')

def selectAction(obs, eps=0.1, policy='MonteCarlo'):
    '''
    In heuristic mode, whenever the angle is positive, it pushes
    the cartpole to right, and when the angle is negative, it
    pushes the cartpole to left.
    '''
    if policy.lower() in ['montecarlo', 'mc']:
        n = np.random.rand()
        if n>eps:
            q = model.predict(np.array([obs]))
            return 0 if q[0, 0] > q[0, 1] else 1
        else:
            return np.random.randint(0, 2)
    elif policy.lower() in ['hand-coded', 'manual', 'heuristic']:
        if obs[2] > 0:
            return 1
        else:
            return 0

eps = 0.2


hist_obs = []
hist_act = []
hist_rew = []
episode_lens = []

for i_episode in range(50):
    hist_obs = []
    hist_act = []
    hist_rew = []

    observation = env.reset()
    hist_obs.append(observation)
    for t in range(200):
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        action = selectAction(observation, eps, 'mc')
        hist_act.append(action)
        observation, reward, done, info = env.step(action)
        hist_rew.append(reward)
        hist_obs.append(observation)
        sleep(0.02)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            findEpisodeStats(hist_obs, hist_act, hist_rew)
            episode_lens.append(t)            
            if not (i_episode % 10):
                saveModel()
            break

print(episode_lens)
plt.figure()
plt.plot(range(len(episode_lens)), episode_lens, 'r--x')
plt.title('Episode Lengths')
plt.show()
env.close()