import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdp_problem.grid_map import GridMap
from model_free.q_learning import QLearning
from model_free.sarsa import SARSA
from multi_armed_bandit.softmax import Softmax
from multi_armed_bandit.upper_confidence_bound import UCB1

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

def plot_cumulative_rewards(cumulative_rewards):
    plt.figure(figsize=(6,4))
    plt.plot(cumulative_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.tight_layout()

if __name__ == "__main__":
    # Learning process
    mdp = GridMap()
    bandit = UCB1()
    planner = QLearning(mdp, bandit)
    # planner = SARSA(mdp, bandit)
    cumulative_rewards = planner.learn(episodes=1000)

    # Execution
    state = (1, 0)
    path = planner.execution(state)
    print("Optimal Path:", path)

    # Plot
    plot_environment_and_path(mdp, path)
    plot_cumulative_rewards(cumulative_rewards)
    plt.show()