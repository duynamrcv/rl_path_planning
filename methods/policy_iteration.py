import numpy as np
from mdp_problem.mdp import MDP
from mdp_problem.grid_map import GridMap

class PolicyIteration():
    def __init__(self, mdp:MDP, gamma=0.9, threshold=1e-4):
        self.mdp = mdp
        self.gamma = gamma
        self.threshold = threshold

        self.action_space = self.mdp.get_actions()
        self.state_size = self.mdp.get_states()
        self.policy = np.random.choice(self.action_space, size=self.state_size)
        self.value = np.zeros(self.state_size)

    def evaluate_policy(self):
        rows, cols = self.state_size
        while True:
            delta = 0
            new_value = self.value.copy()
            for r in range(rows):
                for c in range(cols):
                    state = (r, c)
                    self.mdp.set_state(state)
                    if state == self.mdp.get_goal_states() or not self.mdp.is_valid(state):
                        continue
                    action = self.policy[state]
                    reward, _ = self.mdp.step(action)
                    next_state = self.mdp.get_current_state()
                    new_value[state] = reward + self.gamma * self.value[next_state]
                    delta = max(delta, abs(new_value[state] - self.value[state]))
            self.value = new_value
            if delta < self.threshold:
                break

    def improve_policy(self):
        policy_stable = True
        rows, cols = self.state_size
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                self.mdp.set_state(state)
                if state == self.mdp.get_goal_states() or not self.mdp.is_valid(state):
                    continue
                old_action = self.policy[state]
                action_values = {}
                for action in self.action_space:
                    self.mdp.set_state(state)
                    reward, _ = self.mdp.step(action)
                    next_state = self.mdp.get_current_state()
                    action_values[action] = reward + self.gamma * self.value[next_state]
                self.policy[state] = max(action_values, key=action_values.get)
                if old_action != self.policy[state]:
                    policy_stable = False
        return policy_stable
    
    def policy_iteration(self, verbose=False):
        while True:
            self.evaluate_policy()
            stable = self.improve_policy()
            if stable:
                break
        if verbose:
            print("Optimal Policy:")
            for row in self.policy:
                print(row)

    def execution(self, init, goal, verbose=False):
        self.mdp.set_initial_state(init)
        self.mdp.set_goal_states(goal)
        self.policy_iteration(verbose)

        self.mdp.reset()
        state = self.mdp.get_current_state()
        path = [state]
        while state != self.mdp.get_goal_states():
            action = self.policy[state]
            reward, _ = self.mdp.step(action)
            state = self.mdp.get_current_state()
            path.append(state)
        
        if verbose:
            print("Optimal path:", path)
        return path
