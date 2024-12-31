import math
import random
from mdp_problem.grid_map import GridMap

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, possible_actions):
        return len(self.children) == len(possible_actions)

    def best_child(self, exploration_weight=1.0):
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            exploit = child.value / (child.visits + 1e-6)
            explore = exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
class MCTS():
    def __init__(self, mdp:GridMap, exploration_weight=1.0):
        self.mdp = mdp
        self.exploration_weight = exploration_weight

        # Initialize
        rows, cols = self.mdp.get_states()
        self.action_space = self.mdp.get_actions()

    def get_possible_actions(self, state):
        possible_actions = []
        for action in self.action_space:
            action = self.mdp.action_map[action]
            next_state = (state[0] + action[0], state[1] + action[1])
            if self.mdp.is_valid(next_state):
                possible_actions.append(next_state)
        return possible_actions
    
    def simulate(self, state):
        x, y = state
        goal = self.mdp.get_goal_states()
        distance_to_goal = abs(goal[0] - x) + abs(goal[1] - y)  # norm-1 distance
        return -distance_to_goal  # Negative reward for being far from the goal

    def execution(self, init, goal, episodes=1000):
        if self.mdp.is_valid(init):
            self.mdp.set_initial_state(init)
        else:
            print("[ERROR] Invalid start state")
            exit(0)

        if self.mdp.is_valid(goal):
            self.mdp.set_goal_states(goal)
        else:
            print("[ERROR] Invalid goal state")
            exit(0)

        root = Node(init)
        for _ in range(episodes):
            node = root

            # Selection
            while node.children and node.is_fully_expanded(self.get_possible_actions(node.state)):
                node = node.best_child(self.exploration_weight)

            # Expansion
            if node.state != goal:
                actions = self.get_possible_actions(node.state)
                for action in actions:
                    if action not in [child.state for child in node.children]:
                        child_node = Node(action, parent=node)
                        node.children.append(child_node)
                        node = child_node
                        break
            
            # Simulation
            reward = self.simulate(node.state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        # Extract the best path
        path = [init]
        node = root
        while node.children:
            node = node.best_child(0)  # No exploration weight during path extraction
            path.append(node.state)

        return path