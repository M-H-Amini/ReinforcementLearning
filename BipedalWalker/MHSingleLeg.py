import gym
import numpy as np 
import MHNetworks as mhn
from keras.optimizers import Adam
from keras.models import load_model
import os
import itertools

LOAD_MODEL = True
SAVE_MODEL = True
model_names = 'single_leg_model', 'single_leg_action_model', 'single_leg_episodes'

episode_duration = 1000

def createModelInput(observations, actions):
    # observations[4:8] = observations[4:8] / 10
    # observations[9:13] = observations[9:13] / 10
    # observations[14:23] = observations[14:23] / 10
    return np.concatenate((observations, actions))

def selectAction(model, observation, candidate_actions, eps=0.5):
    n = np.random.rand()
    if n > eps:
        states = [np.concatenate((observation, candidate_actions[i])) for i in range(len(candidate_actions))]
        states = np.array(states)
        qs = model(states)
        action_index = np.argmax(np.squeeze(qs))
        return candidate_actions[action_index]

    print('Random Action!')
    return candidate_actions[np.random.randint(len(candidate_actions))]


env = gym.make('BipedalWalker-v3')
action = np.array([0., 0, -1, 1])

if LOAD_MODEL:
    if os.path.exists(model_names[0]) and os.path.exists(model_names[1]):
        model = load_model(model_names[0])
        action_model = load_model(model_names[1])
        print('Models loaded!')
    else:
        raise OSError('No models saved!!!')
else:
    model = mhn.createModel(24+4, 120)
    model.compile(optimizer=Adam(0.01), loss='mse')

    action_model = mhn.createModel(24+4, 120)
    action_model.compile(optimizer=Adam(0.01), loss='mse')


##  Discretizing actions...
action_no = 8
torques = np.linspace(-1, 1, action_no)
candidate_indexes = []
candidate_actions = []

for i in range(action_no):
    for j in range(action_no):
        candidate_actions.append([torques[i], torques[j], -1, 1])

##  Training...
def plan(model, n, obs_hist, act_hist, rew_hist):
    for _ in range(5):
        for i in range(len(obs_hist)):
            index = np.random.randint(len(act_hist)-1-n)
            # print(len(obs_hist), len(act_hist), len(rew_hist), index)
            state = createModelInput(obs_hist[index], act_hist[index])[np.newaxis, :]
            
            next_state = createModelInput(obs_hist[index+n], act_hist[index+n])[np.newaxis, :]
            bootstrap = model(next_state)[0, 0]
            rewards = sum([rew_hist[index+k] * (0.9**k) for k in range(n)])
            ret = rewards + (0.9**n) * bootstrap

            model.train_on_batch(state, np.array([ret]))

def onlineTrain(model, n, obs_hist, act_hist, rew_hist):
    state = createModelInput(obs_hist[-2-n], act_hist[-1-n])[np.newaxis, :]
    next_state = createModelInput(obs_hist[-2], act_hist[-1])[np.newaxis, :]
    rewards = sum([rew_hist[-1-n+k] * (0.9**k) for k in range(n)])
    bootstrap = model(next_state)[0, 0]
    ret = rewards + (0.9 ** n) * bootstrap

    model.train_on_batch(state, np.array([ret]))


obs_hist = []
act_hist = []
rew_hist = []

episode_lens = []

for i_episode in range(1000000):
    print('Episode ', i_episode+1)
    observation = env.reset()
    obs_hist.clear()
    act_hist.clear()
    rew_hist.clear()
    obs_hist.append(observation)
    for t in range(episode_duration):
        env.render()
        # print(observation)

        if i_episode % 2:
            action = selectAction(action_model, observation, candidate_actions)
        else:
            action = selectAction(model, observation, candidate_actions)
            

        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        act_hist.append(action)
        rew_hist.append(reward)
        obs_hist.append(observation)

        if t > 5:
            if i_episode % 2:
                onlineTrain(model, 5, obs_hist, act_hist, rew_hist)
            else:
                onlineTrain(action_model, 5, obs_hist, act_hist, rew_hist)

        if done or t > episode_duration - 2:
            print("Episode finished after {} timesteps".format(t+1))
            if i_episode % 2:           
                plan(model, 5, obs_hist, act_hist, rew_hist)
            else:
                plan(action_model, 5, obs_hist, act_hist, rew_hist)
                
            episode_lens.append(t)

            if SAVE_MODEL and not(i_episode % 10):
                print('Models saved!!!')
                model.save(model_names[0])
                action_model.save(model_names[1])
                np.save(model_names[2], np.array(episode_lens))
            break
env.close()

print('Episode Lengths: ', episode_lens)