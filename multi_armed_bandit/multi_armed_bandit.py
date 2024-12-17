from abc import abstractmethod

class MultiArmedBandit():
    """ Select an action for this state given from a list given a Q-function
    """
    @abstractmethod
    def select(self, state, action_space, q_table):
        pass

    """ Reset a multi-armed bandit to its initial configuration """
    def reset(self):
        self.__init__()