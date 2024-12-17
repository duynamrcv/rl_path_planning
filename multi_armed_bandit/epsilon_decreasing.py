from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

class EpsilonDecreasing(MultiArmedBandit):
    def __init__(self, epsilon=1.0, alpha=0.999, lower_bound=0.1):
        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.lower_bound = lower_bound

        self.reset()

    def reset(self):
        self.epsilon_greedy_bandit = EpsilonGreedy(self.initial_epsilon)

    def select(self, state, action_space, q_table):
        result = self.epsilon_greedy_bandit.select(state, action_space, q_table)
        self.epsilon_greedy_bandit.epsilon = max(
            self.epsilon_greedy_bandit.epsilon * self.alpha, self.lower_bound
        )
        return result