import numpy as np
from mdp_problem.mdp import MDP
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class QLearning():
    def __init__(self, mdp:MDP, bandit:MultiArmedBandit,
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
        action_space = self.mdp.get_actions()
        return self.bandit.select(state, action_space, self.Q_table)
        
    def learn(self, episodes=2000):
        ''' Train Q-learning agent
        '''
        action_space = self.mdp.get_actions()
        cumulative_rewards = []
        for _ in range(episodes):
            # Reset environment and model
            self.mdp.reset()
            self.bandit.reset()

            done = False
            cumulative_reward = 0
            while not done:
                state = self.mdp.get_current_state()
                action = self.choose_action(state)
                reward, done = self.mdp.step(action)
                next_state = self.mdp.get_current_state()
                # print("Current: {}, Action: {}, Next: {}, Reward: {}".format(state, action, next_state, reward))

                # Q-learning update
                r, c = state
                nr, nc = next_state
                self.Q_table[r, c, action_space.index(action)] += self.lr * (
                    reward + \
                    self.gamma * np.max(self.Q_table[nr, nc]) - \
                    self.Q_table[r, c, action_space.index(action)]
                )
                
                # Update cumulative reward 
                cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards
    
    def execution(self, init, goal, episodes=2000):
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

        # Learn the optimal path
        cumulative_rewards = self.learn(episodes)
        
        # Extract optimal path
        state = init
        action_space = self.mdp.get_actions()
        path = [state]
        while state != goal:
            r, c = state
            # print(state)
            action = action_space[np.argmax(self.Q_table[r, c])]
            dr, dc = self.mdp.action_map[action]
            state = (r + dr, c + dc)
            path.append(state)
        return path, cumulative_rewards