import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches 
import matplotlib
import numpy as np
from time import sleep

class MHGrid:
    def __init__(self, r=3, c=3, initial_pos=(0, 0), target_pos=None, obstacles=None, display=True, delay=0.5):
        self.r = r
        self.c = c
        self.board = np.zeros((r, c))
        
        self.initial_pos = list(initial_pos)
        self.current_pos = self.initial_pos
        self.target_pos = target_pos if target_pos else [r-1, c-1]
        self.obstacles = list()
        if obstacles:
            self.addObstacles(obstacles)
        self.reward = 0
        self.action = 'S'
        self.action_dict = {'L':'Left', 'R':'Right', 'U':'Up', 'D':'Down', 'S':'Stop'}
        
        self.display = display
        if self.display:
            self.fig, self.ax = plt.subplots(figsize=(7,7))
            self.ax.set_title('MHGridWorld', fontsize=20)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            g_patch = mpatches.Patch(facecolor='g', label='Player')
            r_patch = mpatches.Patch(facecolor='r', label='Obstacle')
            y_patch = mpatches.Patch(facecolor='yellow', label='Target')
            plt.legend(handles=[y_patch, g_patch, r_patch],
                     bbox_to_anchor=(0., 1.06, 1., .102))
            self.delay = delay
            self.show(0.01)
        
        self.render()
        self.done = False

    def render(self):
        '''
            Updates the board.
        '''
        self.board[self.target_pos[0], self.target_pos[1]] = 80
        self.board[self.current_pos[0], self.current_pos[1]] = 255
        for pos in self.obstacles:
            self.board[pos[0], pos[1]] = 180
    
    def availableActs(self):
        '''
            Returns available actions in the current position.
        '''
        res = ['L', 'U', 'R', 'D']
        if self.current_pos[0] == 0:
            res.remove('D')
        elif self.current_pos[0] == self.r-1:
            res.remove('U')
        if self.current_pos[1] == 0:
            res.remove('L')
        elif self.current_pos[1] == self.c-1:
            res.remove('R')
        return res

    def act(self, a):
        '''
            Moves the agent and gets a reward.
        '''
        self.action = a
        prev_pos = self.current_pos.copy()
        self.board[self.current_pos[0], self.current_pos[1]] = 0
        out_flag = False  #  Out of borders flag
        obs_flag = False  #  Hitting obstacles flag

        ##  Doing the move...
        if a == 'U':
            self.current_pos[0] = self.current_pos[0] + 1
        elif a == 'D':
            self.current_pos[0] = self.current_pos[0] - 1
        elif a == 'L':
            self.current_pos[1] = self.current_pos[1] - 1
        elif a == 'R':
            self.current_pos[1] = self.current_pos[1] + 1
        
        ##  Checking validity of move...
        if self.current_pos[0] < 0:
            self.current_pos[0] = 0
            out_flag = True
        
        if self.current_pos[0] >= self.r:
            self.current_pos[0] = self.r - 1
            out_flag = True
        
        if self.current_pos[1] < 0:
            self.current_pos[1] = 0
            out_flag = True

        if self.current_pos[1] >= self.c:
            self.current_pos[1] = self.c - 1
            out_flag =True

        if tuple(self.current_pos) in self.obstacles:
            self.current_pos = prev_pos
            obs_flag = True

        ##  Setting reward...
        if out_flag:
            reward = -10
        elif obs_flag:
            reward = -50
        elif self.current_pos == [self.r-1, self.c-1]:
            reward = 1
            self.done = True
            self.winMessage()
        else:
            reward = 0

        self.reward = reward
        if self.display:
            self.show(self.delay)

        return self.current_pos, self.reward, self.done

    def reset(self):
        '''
            Resets the episode.
        '''
        self.board[self.current_pos[0], self.current_pos[1]] = 0
        self.current_pos = [0, 0]
        self.action = 'S'
        self.reward = 0
        self.done = False
        self.render()
        if self.display:
            for text in self.ax.texts:
                text.set_visible(False)
            self.show()

    def getCurrentPosition(self):
        return tuple(self.current_pos)

    def getTargetPosition(self):
        return tuple(self.target_pos)

    def addObstacle(self, pos):
        '''
            Adds a single obstacle.
        '''
        self.obstacles.append(pos)

    def addObstacles(self, pos_list):
        '''
            Adds multiple obstacles. 
            pos_list: a list containing tuples of coordinates for obstacles.
        '''
        for obstacle in pos_list:
            self.addObstacle(obstacle)

    def randomAct(self):
        acts = ['U', 'D', 'R', 'L']
        return acts[np.random.randint(4)]

    def winMessage(self):
        self.ax.text(self.c/2, self.r/2, 'Finished!',
             bbox=dict(facecolor='white', alpha=0.75),
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=52)
        
    def show(self, delay=1):
        self.render()
        plt.pause(delay)
        colormap = matplotlib.colors.ListedColormap(['gray', 'yellow', 'red', 'green'])
        self.ax.pcolor(self.board, edgecolors='k', linewidths=4, cmap=colormap)
        self.ax.set_xlabel(f'Action: {self.action_dict[self.action]:20}  Reward: {self.reward:10}', fontsize=20)
        self.fig.canvas.draw()
    
    def createRandomObstacles(self, n=1):
        obs = []
        while len(obs) < n:
            pos = (np.random.randint(self.r), np.random.randint(self.c))
            if (pos not in obs) and (pos!=(self.r-1,self.c-1)):
                obs.append(pos)

        self.addObstacles(obs)
        if self.display:
            self.show(0.01)
            

if __name__=='__main__':
    # obstacles = [(1,0), (2,0), (2,3)]
    # grid = MHGrid(4, 5, obstacles=obstacles, delay=.3)
    grid = MHGrid(10, 10, delay=.5)
    grid.createRandomObstacles(20)
    for i in range(50):
        a = grid.randomAct()
        grid.act(a)

    # grid.act('R')
    plt.show()