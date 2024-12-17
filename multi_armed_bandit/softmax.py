import math
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class Softmax(MultiArmedBandit):
    def __init__(self, tau=1.0):
        self.tau = tau

    def reset(self):
        pass

    def select(self, state, action_space, q_table):
        r, c = state

        # calculate the denominator for the softmax strategy
        total = 0.0
        for action in action_space:
            total += math.exp(q_table[r, c, action_space.index(action)] / self.tau)

        rand = random.random()
        cumulative_probability = 0.0
        result = None
        for action in action_space:
            probability = (
                math.exp(q_table[r, c, action_space.index(action)] / self.tau) / total
            )
            if cumulative_probability <= rand <= cumulative_probability + probability:
                result = action
            cumulative_probability += probability

        return result