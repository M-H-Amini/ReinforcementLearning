import numpy as np 
import gym
from time import sleep 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

##  mcmodel is trained well for monte carlo

env = gym.make('CartPole-v0')
name = 'qmodel'

def saveModel(name):
    model.save(name)

def loadModel(name):
    model = keras.models.load_model(name)
    return model

# model = loadModel(name)

model = Sequential([
    Dense(6, 'tanh', input_shape=(4,)),
    Dense(2)
])

model_train = Sequential([
    model,
    Dense(1)
])

model_train.compile(optimizer=Adam(0.01), loss='mse')

model_target = keras.models.clone_model(model)

def copyModel(target, source):
    target.set_weights(source.get_weights())
    return target

def processModel(model, i):
    w = np.zeros((2, 1))
    w[i, 0] = 1
    model.layers[-1].set_weights([w, np.array([0])])
    model.layers[-1].trainable = False

def qlearning(n, lamda, t, model_train, model, hist_obs, hist_act, hist_rew):
    '''
    t: current episode time, started from 0
    '''
    if t+1 >= n:
        observed_returns = sum([hist_rew[-(i+1)]*(lamda**i) for i in range(n)])
        ret = observed_returns + (lamda**n) * np.max(model(np.array([hist_obs[-1]]))[0, :])
        X = np.array([hist_obs[-(n+1)]])
        y = np.array([ret])
        # print(f'hist_rew[-n:]: {hist_rew[-n:]}\tBoot: {model(np.array([hist_obs[-1]]))}')
        # print(f'Return: {ret}')
        # print('X: ', X, X.shape)
        # print('y: ', y, y.shape)
        processModel(model_train, hist_act[-n])
        for _ in range(2):
            model_train.train_on_batch(X, y)


def selectAction(model, obs, eps=0.1, policy='q'):
    '''
    In heuristic mode, whenever the angle is positive, it pushes
    the cartpole to right, and when the angle is negative, it
    pushes the cartpole to left.
    '''
    if policy.lower() in ['q', 'qlearning']:
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

done_t = 0
done_len = 10
done_flag = True
eps = 0.3


hist_obs = []
hist_act = []
hist_rew = []
episode_lens = []

for i_episode in range(1000):
    print('-'*20, 'new episode!')
    done_flag = False
    hist_obs = []
    hist_act = []
    hist_rew = []

    observation = env.reset()
    hist_obs.append(observation)
    for t in range(200):
        env.render()
        action = selectAction(model_target, observation, eps, 'q')
        hist_act.append(action)
        observation, reward, done, info = env.step(action)
        hist_rew.append(reward)
        hist_obs.append(observation)
        qlearning(3, 0.8, t ,model_train, model, hist_obs, hist_act, hist_rew)
        sleep(0.01)
        if done:
            if done_flag == False:
                done_flag = True
                done_t = 0
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                episode_lens.append(t)            
            else:
                done_t += 1
        #     if not (i_episode % 10):
        #         saveModel(name)
        #     if done_t > done_len:
        #         done_flag = False
        #         break
    
    if not (i_episode % 3):
                    model_target = copyModel(model_target, model)
                    print("Models copied!!!")

print(episode_lens)
plt.figure()
plt.plot(range(len(episode_lens)), episode_lens, 'r--x')
plt.title('Episode Lengths')
plt.show()
env.close()