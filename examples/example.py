from mdp_problem.grid_map import GridMap
from model_free.q_learning import QLearning

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Learning process
    mdp = GridMap()
    planner = QLearning(mdp)
    cumulative_rewards = planner.learn(episodes=1000)

    # Execution
    state = (1, 2)
    path = planner.execution(state)
    print("Optimal Path:", path)

    # Plot
    plt.figure()
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.tight_layout()
    plt.show()