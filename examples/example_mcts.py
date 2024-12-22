import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdp_problem.grid_map import GridMap
from model_free.mcts import MCTS
import numpy as np
import matplotlib.pyplot as plt

def plot_environment_and_path(mdp:GridMap, path):
    # Plot the environment
    plt.figure(figsize=(6, 4))
    environment = np.zeros([mdp.rows, mdp.cols])
    for (r,c) in mdp.obstacles:
        environment[r, c] = 1
    plt.imshow(environment, cmap="gray_r", origin="upper")

    # Plot path
    start = path[0]
    goal = path[-1]
    path = np.array(path)   # Path
    plt.plot(path[:,1], path[:,0], linewidth=5)
    plt.scatter(start[1], start[0], color="green", s=100, label="Start", zorder=2)  # Start point
    plt.scatter(goal[1], goal[0], color="red", s=100, label="Goal", zorder=2)      # Goal point

    # Customize the grid
    plt.xticks(range(environment.shape[1]))
    plt.yticks(range(environment.shape[0]))
    plt.grid(visible=True, color="black", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    # Learning process
    mdp = GridMap()
    planner = MCTS(mdp, exploration_weight=0.1)

    # Execution
    state = (3, 2)
    goal = (4, 7)
    path = planner.execution(state, goal, episodes=1000)
    print("Optimal Path:", path)

    # Plot
    plot_environment_and_path(mdp, path)
    plt.show()