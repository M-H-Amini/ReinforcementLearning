import numpy as np 
from itertools import permutations
import time

class Puzzle:
    def __init__(self, dim=2):
        self.dim = dim
        self.p = [[j * self.dim + i for i in range(self.dim)] for j in range(self.dim)]
        print(self.p)
        self.zero = self.findZero()
        l = [''.join([str(j) for j in i]) for i in list(permutations(range(self.dim**2)))]        
        self.q_dict={key:{a:0 for a in range(4)} for key in l}
        self.act_dict = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}

    def findZero(self):
        for i, v in enumerate(self.p):
            if 0 in v:
                return (i, v.index(0))
    
    def show(self):
        print('Puzzle: ')
        for i in range(self.dim):
            print(' '*10, self.p[i])

    def act(self, dir):
        z = self.findZero()
        if dir==0 and z[0]!=0:
            self.p[z[0]][z[1]], self.p[z[0]-1][z[1]] = self.p[z[0]-1][z[1]], self.p[z[0]][z[1]]
        elif dir==2 and z[0]!=self.dim-1:
            self.p[z[0]][z[1]], self.p[z[0]+1][z[1]] = self.p[z[0]+1][z[1]], self.p[z[0]][z[1]]
        elif dir==1 and z[1]!=self.dim-1:
            self.p[z[0]][z[1]], self.p[z[0]][z[1]+1] = self.p[z[0]][z[1]+1], self.p[z[0]][z[1]]
        elif dir==3 and z[1]!=0:
            self.p[z[0]][z[1]], self.p[z[0]][z[1]-1] = self.p[z[0]][z[1]-1], self.p[z[0]][z[1]]
        return 1 if self.isDone() else 0
    
    def optimalAct(self):
        legal_acts = self.findLegalActs()
        q_s = [self.q_dict[self.puzzle2str()][a] for a in legal_acts]
        max_q = max(q_s)
        max_as = [a for a in legal_acts if self.q_dict[self.puzzle2str()][a]==max_q]
        return max_as[np.random.randint(0, len(max_as))]

    def isActOK(self, a):
        z = self.findZero()
        if z[0]==0 and a==0:
            return False
        elif z[0]==self.dim-1 and a==2:
            return False
        elif z[1]==0 and a==3:
            return False
        elif z[1]==self.dim-1 and a==1:
            return False
        return True

    def findLegalActs(self):
        return [i for i in range(4) if self.isActOK(i)]

    def isDone(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.p[i][j]!=i*self.dim + j:
                    return False
        return True

    def policy(self, eps=0.2):
        n = np.random.rand()
        if n<eps:
            legal_acts = self.findLegalActs()
            return legal_acts[np.random.randint(0, len(legal_acts))]
        return self.optimalAct() 

    def puzzle2str(self):
        res = ''
        for i in range(self.dim):        
            for j in range(self.dim):
                res += str(self.p[i][j])
        return res
    
    def shuffle(self):
        while self.isDone():
            for i in range(100):
                self.act(np.random.randint(0, 4))
        return self.p

    def resetState(self):
        self.p = [[j * self.dim + i for i in range(self.dim)] for j in range(self.dim)]
        return self.p

    def play(self):
        print('-'*20, 'Playing', '-'*20)
        no = 0
        while not self.isDone():
            a = self.optimalAct()
            print(f'T={no}, A={self.act_dict[a]}:')
            self.show()
            self.act(a)
            no += 1
        print(f'T={no}')
        self.show()


def MonteCarlo(puzzle, state_history, act_history, reward_history, N_dict, disc=1, first_visit=True):
    for i in range(len(state_history)-2, -1, -1):
        ret = 0
        last_indexes = [index for index in range(i) if state_history[index]==state_history[i]]
        last_actions = [act_history[index] for index in last_indexes]

        if (first_visit and (len(last_indexes)==0 or not act_history[i] in last_actions)) or (not first_visit):
            if disc==1:
                ret = sum(reward_history[i+1:])
            else:
                for j in range(i+1, len(reward_history)):
                    ret += reward_history[j] * disc**(j-(i+1))
            N = N_dict[state_history[i]][act_history[i]]
            puzzle.q_dict[state_history[i]][act_history[i]] = (N * puzzle.q_dict[state_history[i]][act_history[i]] + ret) / (N + 1)
            N_dict[state_history[i]][act_history[i]] = N + 1
    return puzzle.q_dict, N_dict


def generateEpisodes(puzzle, episodes=1000, verbose=1):
    act_dict = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
    N_dict = {key:{a:0 for a in range(4)} for key in puzzle.q_dict.keys()}
    for e in range(episodes):
        puzzle.shuffle()
        #puzzle.resetState()
        act_history = []
        state_history = []
        reward_history = []
        ret = 0
        if verbose==2:
            print(f'Episode {e+1}:')
        t = 0
        a = np.random.randint(0, 2) + 1
        if verbose==2:
            print('T= {}, Act: {}'.format(t, act_dict[a]))
        reward = puzzle.act(a)
        ret += reward
        state_history.append(puzzle.puzzle2str())
        reward_history.append(reward)
        if verbose==2:
            puzzle.show()

        while True:
            t += 1
            a = puzzle.policy(eps=0.4)
            while t>5 and abs(a-act_history[-1])==2 and a==act_history[-2] and abs(a-act_history[-3])==2:
                a = puzzle.policy()
            reward = puzzle.act(a)
            ret += reward
            if verbose==2:
                print('E= {}, T= {}, Act: {}'.format(e, t, act_dict[a]))
            act_history.append(a)
            state_history.append(puzzle.puzzle2str())
            reward_history.append(reward)
            if verbose==2:
                puzzle.show()
            if puzzle.isDone():
                if verbose==2:
                    print('Acts: ', len(act_history), [act_dict[act] for act in act_history])
                    print('States: ', len(state_history), state_history)
                    print('Rewards: ', len(reward_history), reward_history)
                if verbose==1:
                    print(f'Episode {e+1}: Duration {t} time steps')
                puzzle.q_dict, N_dict = MonteCarlo(puzzle, state_history, act_history, reward_history, N_dict, disc=0.9, first_visit=False)
                break
        
    return puzzle
            
        
if __name__=='__main__':        
    my_puzzle = Puzzle(2)
    t0 = time.time()
    my_puzzle = generateEpisodes(my_puzzle, 10, verbose=2)
    t1 = time.time()
    print('Took {} seconds'.format(t1-t0))
    print('Test Time: ')
    for i in range(10):
        print('*'*20, f'Test {i+1} of {10}', '*'*20)
        my_puzzle.shuffle()
        my_puzzle.play()
