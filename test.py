import math
import random

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

def mcts_path_planning(start, goal, grid, max_iterations=1000, exploration_weight=1.0):
    root = Node(start)

    def is_goal(state):
        return state == goal

    def get_possible_actions(state):
        actions = []
        x, y = state
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy in moves:
            next_state = (x + dx, y + dy)
            if 0 <= next_state[0] < len(grid) and 0 <= next_state[1] < len(grid[0]) and grid[next_state[0]][next_state[1]] == 0:
                actions.append(next_state)
        return actions

    def simulate(state):
        x, y = state
        distance_to_goal = abs(goal[0] - x) + abs(goal[1] - y)
        return -distance_to_goal  # Negative reward for being far from the goal

    for _ in range(max_iterations):
        node = root

        # Selection
        while node.children and node.is_fully_expanded(get_possible_actions(node.state)):
            node = node.best_child(exploration_weight)

        # Expansion
        if not is_goal(node.state):
            actions = get_possible_actions(node.state)
            for action in actions:
                if action not in [child.state for child in node.children]:
                    child_node = Node(action, parent=node)
                    node.children.append(child_node)
                    node = child_node
                    break

        # Simulation
        reward = simulate(node.state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Extract the best path
    path = []
    node = root
    while node.children:
        node = node.best_child(0)  # No exploration weight during path extraction
        path.append(node.state)

    return path

# Example usage
grid = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
]
start = (3, 0)
goal = (3, 3)
path = mcts_path_planning(start, goal, grid)
print("Optimal Path:", path)
