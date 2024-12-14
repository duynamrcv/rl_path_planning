import numpy as np
import random
import matplotlib.pyplot as plt

from env import Env

class QLearning:
    def __init__(self, env:Env, learning_rate=0.01, reward_decay=0.9, explore_rate=0.9):
        self.env = env
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = explore_rate

        # Initialize Q-table
        self.Q_table = np.zeros((self.env.rows,
                                 self.env.cols,
                                 len(self.env.action_space)))

    def choose_action(self, state):
        ''' Choose action (epsilon-greedy)
        '''
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)  # Explore
        else:
            row, col = state
            return self.env.action_space[np.argmax(self.Q_table[row, col])]  # Exploit
        
    def learn(self, num_episodes=1000):
        ''' Train Q-learning agent
        '''
        cumulative_rewards = []
        for _ in range(num_episodes):
            self.env.reset()
            done = False
            cumulative_reward = 0
            while not done:
                r, c = self.env.agent
                action = self.choose_action(self.env.agent)
                reward, done = self.env.step(action)
                nr, nc = self.env.agent

                # Q-learning update
                self.Q_table[r, c, self.env.action_space.index(action)] += self.lr * (
                    reward + \
                    self.gamma * np.max(self.Q_table[nr, nc]) - \
                    self.Q_table[r, c, self.env.action_space.index(action)]
                )
                
                # Update cumulative reward 
                cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards
    
    def execution(self, state):
        # Extract optimal path
        path = [state]
        while state != self.env.goal:
            r, c = state
            action = self.env.action_space[np.argmax(self.Q_table[r, c])]
            dr, dc = self.env.action_map[action]
            state = (r + dr, c + dc)
            path.append(state)
        return path

if __name__ == "__main__":
    # Learning process
    env = Env()
    planner = QLearning(env)
    cumulative_rewards = planner.learn(num_episodes=1000)

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