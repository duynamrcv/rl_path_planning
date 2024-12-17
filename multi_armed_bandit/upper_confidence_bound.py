import math
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class UCB1(MultiArmedBandit):
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, state, action_space, q_table):

        # First execute each action one time
        for action in action_space:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_action_space = []
        max_value = float("-inf")
        r, c = state
        for action in action_space:
            value = q_table[r, c, action_space.index(action)] + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_action_space = [action]
                max_value = value
            elif value == max_value:
                max_action_space += [action]

        # if there are multiple action_space with the highest value
        # choose one randomly
        result = random.choice(max_action_space)
        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1
        return result