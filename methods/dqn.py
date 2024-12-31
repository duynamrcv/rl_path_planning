import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from mdp_problem.mdp import MDP

# Neural network for Q-value approximation
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, mdp:MDP, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)

        # Neural networks
        self.rows, self.cols = self.mdp.get_states()
        self.actions = self.mdp.get_actions()
        self.model = DQNetwork(self.rows*self.cols, len(self.actions))
        self.target_model = DQNetwork(self.rows*self.cols, len(self.actions))
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        # Using epsilon greedy
        if random.uniform(0, 1) < self.epsilon:

            return random.randint(0, len(self.actions) - 1)  # Explore
        else:
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_tensor = self.state_to_tensor(state)
            next_state_tensor = self.state_to_tensor(next_state)
            target = self.model(state_tensor.unsqueeze(0))
            target_value = reward
            if not done:
                with torch.no_grad():
                    target_value += self.gamma * torch.max(self.target_model(next_state_tensor.unsqueeze(0)))
            target[0][action] = target_value

            # Compute loss and optimize
            self.optimizer.zero_grad()
            output = self.model(state_tensor.unsqueeze(0))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Convert state to a one-hot vector
    def state_to_tensor(self, state):
        grid = np.zeros((self.rows, self.cols))
        grid[state] = 1
        return torch.tensor(grid.flatten(), dtype=torch.float32).to(self.model.device)
    
    def learn(self, episodes, batch_size):
        # Training loop
        total_rewards = []
        for e in range(episodes):
            # Reset environment and model
            self.mdp.reset()
            state = self.mdp.get_current_state()
            total_reward = 0
            for time_step in range(200):  # Limit steps per episode
                action = self.act(state)
                reward, done = self.mdp.step(self.actions[action])
                next_state = self.mdp.get_current_state()
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode {e + 1}/{episodes}: Reward: {total_reward}, Steps: {time_step + 1}")
                    break

                self.replay(batch_size)
            total_rewards.append(total_reward)
            self.reduce_epsilon()
            self.update_target_model()
        return np.array(total_rewards)

    def save_models(self, episode, directory="models"):
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model.state_dict(), f"{directory}/dqn_model_{episode}.pth")
        torch.save(self.target_model.state_dict(), f"{directory}/dqn_target_model_{episode}.pth")
        print(f"Models saved for episode {episode}")

    def execution(self, init, goal, episodes=2000):
        # Check input - output
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

        # Learn and save the model
        total_rewards = self.learn(episodes=episodes, batch_size=32)
        id_sort = np.argsort(total_rewards)
        self.save_models(id_sort[-1])

        # Extract optimal path
        self.mdp.reset()
        state = self.mdp.get_current_state()
        done = False
        path = [state]
        while not done:
            action = self.act(state)
            _, done = self.mdp.step(self.actions[action])
            state = self.mdp.get_current_state()
            path.append(state)
        return path, total_rewards
