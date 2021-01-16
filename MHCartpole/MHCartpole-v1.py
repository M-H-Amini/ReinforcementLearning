import numpy as np 
import gym
from time import sleep 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def saveModel():
    model.save('model')

def loadModel():
    model = keras.models.load_model('model')
    return model

model = loadModel()

# model = Sequential([
#     Dense(6, 'tanh', input_shape=(4,)),
#     # Dense(2, 'tanh'),
#     Dense(2)
# ])

model_train = Sequential([
    model,
    Dense(1)
])

model_train.compile(optimizer=Adam(0.03), loss='mse')

model_target = keras.models.clone_model(model)

def copyModel(target, source):
    target.set_weights(source.get_weights())
    return target

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


def findEpisodeStats(model_train, model, hist_obs, hist_act, hist_rew, verbose=False):
    if verbose:
        print('Before Training: ')
    X_batch, y_batch = [list() for i in range(2)], [list() for i in range(2)]
    for i in range(min(len(hist_act), 100)):
        ret = sum(hist_rew[i:])
        X_batch[hist_act[i]].append(hist_obs[i])
        y_batch[hist_act[i]].append(ret)
        if verbose:
            print(f'{i}: ret: {ret}, a: {hist_act[i]}\tPredicted:{model(np.array([hist_obs[i]]))}')

    X_batch = [np.array(i) for i in X_batch]
    y_batch = [np.array(i) for i in y_batch]

    for i in range(2):
        if len(X_batch[i]):
            processModel(model_train, i)
            for _ in range(10):
                model_train.train_on_batch(X_batch[i], y_batch[i])      

    if verbose:
        print('After Training: ')
    for i in range(len(hist_act)):
        ret = sum(hist_rew[i:])
        if verbose:
            print(f'{i}: ret: {ret}, a: {hist_act[i]}\tPredicted:{model(np.array([hist_obs[i]]))}')      
    return model_train

def selectAction(model, obs, eps=0.1, policy='MonteCarlo'):
    '''
    In heuristic mode, whenever the angle is positive, it pushes
    the cartpole to right, and when the angle is negative, it
    pushes the cartpole to left.
    '''
    if policy.lower() in ['montecarlo', 'mc']:
        n = np.random.rand()
        if n>eps:
            q = model.predict(np.array([obs]))
            return np.argmax(q[0, :])
        else:
            action = np.random.randint(0, 2)
            return action
    
    elif policy.lower() in ['hand-coded', 'manual', 'heuristic']:
        if obs[2] > 0:
            return 1
        else:
            return 0

eps = 0.3


hist_obs = []
hist_act = []
hist_rew = []
episode_lens = []

for i_episode in range(10000):
    hist_obs = []
    hist_act = []
    hist_rew = []

    observation = env.reset()
    hist_obs.append(observation)
    for t in range(200):
        env.render()
        action = selectAction(model_target, observation, eps, 'mc')
        hist_act.append(action)
        observation, reward, done, info = env.step(action)
        hist_rew.append(reward)
        hist_obs.append(observation)
        sleep(0.01)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            model_train = findEpisodeStats(model_train, model, hist_obs, hist_act, hist_rew)
            episode_lens.append(t)            
            if not (i_episode % 10):
                saveModel()
            if not (i_episode % 2):
                model_target = copyModel(model_target, model)
                print("Models copied!!!")
            break

print(episode_lens)
plt.figure()
plt.plot(range(len(episode_lens)), episode_lens, 'r--x')
plt.title('Episode Lengths')
plt.show()
env.close()