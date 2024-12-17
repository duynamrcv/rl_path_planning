import numpy as np
# from mdp_problem.mdp import MDP

class GridMap():
    def __init__(self):
        self.action_space = ['up', 'down', 'left', 'right']
        self.action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.create_environment()
        self.reset()

    def create_environment(self):
        ''' Environment configuration
        '''
        self.rows = 5
        self.cols = 8
        self.goal = (4, 7)
        self.obstacles = [(0, 4), (1, 1), (1, 2), (1, 4), (1, 6), (2, 2), (3, 0), (3, 4), (3, 5), (3, 7), (4, 2), (4, 5)]

    def reset(self):
        ''' Reset environment and agent
        '''
        self.agent = (0, 0)
        
    def is_valid(self, state):
        ''' Check valid state
        '''
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and state not in self.obstacles

    def step(self, action):
        ''' Compute the reward funtion and update agent's state
        '''
        r, c = self.agent
        dr, dc = self.action_map[action]
        next_state = (r + dr, c + dc)

        if self.is_valid(next_state):
            self.agent = next_state # Update state

        # Compute the reward
        if self.agent == self.goal:
            reward = 100
            done = True
        elif self.agent in self.obstacles:
            reward = -100
            done = True
        else:
            reward = -1
            done = False
        return reward, done