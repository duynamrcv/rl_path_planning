import numpy as np
from mdp_problem.mdp import MDP
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class SARSA():
    def __init__(self, mdp:MDP, bandit:MultiArmedBandit,
                 learning_rate=0.01, reward_decay=0.9):
        self.mdp = mdp
        self.bandit = bandit

        self.lr = learning_rate
        self.gamma = reward_decay

        # Initialize Q-table
        rows, cols = self.mdp.get_states()
        self.action_space = self.mdp.get_actions()
        self.Q_table = np.zeros((rows, cols, len(self.action_space)))

    def choose_action(self, state):
        ''' Choose action (epsilon-greedy)
        '''
        return self.bandit.select(state, self.action_space, self.Q_table)
        
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
            state = self.mdp.get_current_state()
            action = self.choose_action(state)
            while not done:
                reward, done = self.mdp.step(action)
                
                # Choose next action
                next_state =  self.mdp.get_current_state()
                next_action = self.choose_action(next_state)
                # print("Current: {}, Action: {}, Reward: {}, Next state: {}, Next action: {}".format(state, action, reward, next_state, next_action))

                # SARSA update
                r, c = state
                nr, nc = next_state
                self.Q_table[r, c, self.action_space.index(action)] += self.lr * (
                    reward + \
                    self.gamma * self.Q_table[nr, nc, self.action_space.index(next_action)] - \
                    self.Q_table[r, c, self.action_space.index(action)]
                )

                # Move to next action
                state = next_state
                action = next_action
                
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
        self.mdp.reset()
        state = self.mdp.get_current_state()
        done = False
        path = [state]
        while not done:
            r, c = state
            action = self.action_space[np.argmax(self.Q_table[r, c])]
            _, done = self.mdp.step(action)
            state = self.mdp.get_current_state()
            path.append(state)
        return path, cumulative_rewards