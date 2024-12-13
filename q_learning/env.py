import numpy as np
import time

class Env():
    def __init__(self):
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)

        # Show the steps for shortest route
        self.shortest_path = []
        self.reset()

    def reset(self):
        # Create map with obstacles
        self.x_dim = 10     # X dimension
        self.y_dim = 10     # Y dimension 
        self.env = np.zeros([self.y_dim, self.x_dim])   # 0 - free, 1 - obstacle
        obs = (np.array([1,1,1,3,4,5,0,1,2,3,4,5,6,7,8,1,1]),
               np.array([1,2,3,3,3,3,6,6,6,6,6,6,6,6,6,8,9]))
        self.env[obs] = 1

        # Final target
        self.goal = np.array([0,9])

        # Current agent
        self.agent = np.array([0,0])

        # Save path
        self.path = [self.agent]

    def step(self, action):
        # Moving agent
        if action == 0 and self.agent[1] >= 1:                  # up
            self.agent[1] = self.agent[1] - 1
        elif action == 1 and self.agent[1] <= self.x_dim - 2:   # down
            self.agent[1] = self.agent[1] + 1
        elif action == 2 and self.agent[0] >= 1:                # left
            self.agent[0] = self.agent[0] - 1
        elif action == 3 and self.agent[0] <= self.y_dim - 2:   # right
            self.agent[0] = self.agent[0] + 1

        # Store the path
        self.path.append(self.agent)

        # Compute the reward
        if self.agent[0] == self.goal[0] and self.agent[1] == self.goal[1]: # Reach target
            reward = 1
            done = True
        elif self.env[self.agent[0], self.agent[1]] == 1: # Collision
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return reward, done

    def render(self):
        self.update()

    def final(self):
        ...

    