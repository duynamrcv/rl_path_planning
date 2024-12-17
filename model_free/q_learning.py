import numpy as np
import random
from mdp_problem.grid_map import GridMap

class QLearning():
    def __init__(self, mdp:GridMap, learning_rate=0.01, reward_decay=0.9, explore_rate=0.9):
        self.mdp = mdp

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = explore_rate

        # Initialize Q-table    
        self.Q_table = np.zeros((self.mdp.rows,
                                 self.mdp.cols,
                                 len(self.mdp.action_space)))

    def choose_action(self, state):
        ''' Choose action (epsilon-greedy)
        '''
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.mdp.action_space)  # Explore
        else:
            row, col = state
            return self.mdp.action_space[np.argmax(self.Q_table[row, col])]  # Exploit
        
    def learn(self, episodes=2000):
        ''' Train Q-learning agent
        '''
        cumulative_rewards = []
        for _ in range(episodes):
            self.mdp.reset()
            done = False
            cumulative_reward = 0
            while not done:
                r, c = self.mdp.agent
                action = self.choose_action(self.mdp.agent)
                reward, done = self.mdp.step(action)
                nr, nc = self.mdp.agent

                # Q-learning update
                self.Q_table[r, c, self.mdp.action_space.index(action)] += self.lr * (
                    reward + \
                    self.gamma * np.max(self.Q_table[nr, nc]) - \
                    self.Q_table[r, c, self.mdp.action_space.index(action)]
                )
                
                # Update cumulative reward 
                cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards
    
    def execution(self, state):
        # Extract optimal path
        path = [state]
        while state != self.mdp.goal:
            r, c = state
            action = self.mdp.action_space[np.argmax(self.Q_table[r, c])]
            dr, dc = self.mdp.action_map[action]
            state = (r + dr, c + dc)
            path.append(state)
        return path