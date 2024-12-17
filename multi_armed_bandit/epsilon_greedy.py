import numpy as np
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def reset(self):
        pass

    def select(self, state, action_space, q_table):
        # Select a random action with epsilon probability
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(action_space)  # Explore
        else:
            row, col = state
            return action_space[np.argmax(q_table[row, col])]  # Exploit