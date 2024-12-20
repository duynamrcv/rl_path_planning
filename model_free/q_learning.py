import numpy as np
from mdp_problem.grid_map import GridMap
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class QLearning():
    def __init__(self, mdp:GridMap, bandit:MultiArmedBandit,
                 learning_rate=0.01, reward_decay=0.9):
        self.mdp = mdp
        self.bandit = bandit

        self.lr = learning_rate
        self.gamma = reward_decay

        # Initialize Q-table
        rows, cols = self.mdp.get_states()
        action_space = self.mdp.get_actions()
        self.Q_table = np.zeros((rows, cols,
                                 len(action_space)))

    def choose_action(self, state):
        ''' Choose action (epsilon-greedy)
        '''
        return self.bandit.select(state, self.mdp.action_space, self.Q_table)
        
    def learn(self, episodes=2000):
        ''' Train Q-learning agent
        '''
        cumulative_rewards = []
        for _ in range(episodes):
            # Reset environment and model
            self.mdp.reset()
            self.bandit.reset()

            done = False
            cumulative_reward = 0
            while not done:
                state = self.mdp.agent
                action = self.choose_action(self.mdp.agent)
                self.mdp.step(action)
                next_state = self.mdp.agent
                reward, done = self.mdp.get_reward(state, action, next_state)

                # Q-learning update
                r, c = state
                nr, nc = next_state
                self.Q_table[r, c, self.mdp.action_space.index(action)] += self.lr * (
                    reward + \
                    self.gamma * np.max(self.Q_table[nr, nc]) - \
                    self.Q_table[r, c, self.mdp.action_space.index(action)]
                )
                
                # Update cumulative reward 
                cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards
    
    def execution(self, init, goal):
        self.mdp.set_initial_state(init)
        self.mdp.set_goal_states(goal)
        ...
        # Extract optimal path
        path = [state]
        while state != self.mdp.goal:
            r, c = state
            action = self.mdp.action_space[np.argmax(self.Q_table[r, c])]
            dr, dc = self.mdp.action_map[action]
            state = (r + dr, c + dc)
            path.append(state)
        return path