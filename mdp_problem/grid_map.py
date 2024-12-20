import numpy as np
from mdp_problem.mdp import MDP

class GridMap(MDP):
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
        # self.goal = (4, 7)
        self.agent = None
        self.goal = None
        self.obstacles = [(0, 4), (1, 1), (1, 2), (1, 4), (1, 6), (2, 2), (3, 0), (3, 4), (3, 5), (3, 7), (4, 2), (4, 5)]

    def reset(self):
        ''' Reset environment and agent
        '''
        # self.agent = (0, 0)
        self.agent = self.get_initial_state()
        self.goal = self.get_goal_states()
        
    def is_valid(self, state):
        ''' Check valid state
        '''
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and state not in self.obstacles

    def step(self, action):
        ''' Compute the next state funtion and update agent's state
        '''
        state = self.agent
        action = self.action_map[action]
        next_state = (state[0] + action[0], state[1] + action[1])

        # update agent
        self.agent = next_state
        return state, action, next_state

    def get_reward(self, state, action, next_state):
        # Compute the reward
        if self.agent == self.goal:
            reward = 100
            done = True
        elif self.is_valid(next_state):
            reward = -100
            done = True
        else:
            reward = -1
            done = False
        return reward, done
    
    def set_initial_state(self, state):
        self.agent = state

    def set_goal_states(self, goal):
        self.goal = goal

    def get_initial_state(self):
        return self.agent

    def get_goal_states(self):
        return self.goal
    
    def get_states(self):
        return self.rows, self.cols

    def get_actions(self):
        return self.action_space