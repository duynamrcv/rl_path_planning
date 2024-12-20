from abc import abstractmethod

class MDP():
    @abstractmethod
    def get_states(self):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_valid(self, state):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_goal_states(self):
        pass

    @abstractmethod
    def set_initial_state(self, state):
        pass

    @abstractmethod
    def set_goal_states(self, goal):
        pass