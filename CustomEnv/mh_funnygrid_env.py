import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MHFunnyGrid(gym.Env):
    meta_data = {'render_modes': ['human', 'ansi']}
    def __init__(self, render_mode=None, max_steps=100):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.current_pos = 0
        self.action_value = {0: -1, 1: 1}
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.window = None

    def getInfo(self):
        return {'render_mode': self.render_mode, 'max_steps': self.max_steps}
    
    def getObs(self):
        return self.current_pos 
    
    def reset(self):
        super().reset()
        self.current_pos = 0
        observation = self.getObs()
        info = self.getInfo()
        return observation, info
    
    def step(self, action):
        if self.current_pos != 1:
            self.current_pos += self.action_value[action]
        else:
            self.current_pos -= self.action_value[action]
        self.current_pos = max(0, self.current_pos)
        done = self.current_pos == 3 
        reward = -1
        info = self.getInfo()
        observation = self.getObs()
        return observation, reward, done, False, info
    
if __name__ == '__main__':
    env = MHFunnyGrid()


