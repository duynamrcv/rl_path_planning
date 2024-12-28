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
        agent = DQNAgent(state_size=self.rows*self.cols, action_size=len(self.actions))
        for e in range(episodes):
            # Reset environment and model
            self.mdp.reset()
            state = self.mdp.get_current_state()
            total_reward = 0
            for time_step in range(200):  # Limit steps per episode
                action = agent.act(state)
                reward, done = self.mdp.step(self.actions[action])
                next_state = self.mdp.get_current_state()
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode {e + 1}/{episodes}: Reward: {total_reward}, Steps: {time_step + 1}")
                    break

                agent.replay(batch_size)
            agent.reduce_epsilon()
            agent.update_target_model()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# from collections import deque

# # Grid environment configuration
# grid_size = (5, 8)
# start_state = (1, 0)
# goal_state = (4, 7)
# obstacles = [(0, 4), (1, 1), (1, 2), (1, 4), (1, 6), (2, 2), (3, 0), (3, 4), (3, 5), (3, 7), (4, 2), (4, 5)]
# actions = ['up', 'down', 'left', 'right']
# action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# # Check if a state is valid
# def is_valid(state):
#     r, c = state
#     return 0 <= r < grid_size[0] and 0 <= c < grid_size[1] and state not in obstacles

# # Apply an action to a state
# def take_action(state, action):
#     r, c = state
#     dr, dc = action_map[action]
#     new_state = (r + dr, c + dc)
#     return new_state if is_valid(new_state) else state

# # Reward function
# def reward(state):
#     if state == goal_state:
#         return 100
#     elif state in obstacles:
#         return -100
#     else:
#         return -1

# # Convert state to a one-hot vector
# def state_to_tensor(state, device):
#     grid = np.zeros(grid_size)
#     grid[state] = 1
#     return torch.tensor(grid.flatten(), dtype=torch.float32).to(device)

# # Neural network for Q-value approximation
# class DQNetwork(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_size)

#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # DQN agent
# class DQNAgent:
#     def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#         self.memory = deque(maxlen=2000)

#         # Neural networks
#         self.model = DQNetwork(state_size, action_size)
#         self.target_model = DQNetwork(state_size, action_size)
#         self.update_target_model()

#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def act(self, state):
#         if random.uniform(0, 1) < self.epsilon:
#             return random.randint(0, self.action_size - 1)  # Explore
#         else:
#             state_tensor = state_to_tensor(state, self.model.device).unsqueeze(0)
#             with torch.no_grad():
#                 q_values = self.model(state_tensor)
#             return torch.argmax(q_values).item()  # Exploit

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return
#         batch = random.sample(self.memory, batch_size)

#         for state, action, reward, next_state, done in batch:
#             state_tensor = state_to_tensor(state, self.model.device)
#             next_state_tensor = state_to_tensor(next_state, self.model.device)
#             target = self.model(state_tensor.unsqueeze(0))
#             target_value = reward
#             if not done:
#                 with torch.no_grad():
#                     target_value += self.gamma * torch.max(self.target_model(next_state_tensor.unsqueeze(0)))
#             target[0][action] = target_value

#             # Compute loss and optimize
#             self.optimizer.zero_grad()
#             output = self.model(state_tensor.unsqueeze(0))
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#     def reduce_epsilon(self):
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# # Training loop
# def train_dqn(episodes, batch_size):
#     agent = DQNAgent(state_size=grid_size[0] * grid_size[1], action_size=len(actions))
#     for e in range(episodes):
#         state = start_state
#         total_reward = 0
#         for time_step in range(200):  # Limit steps per episode
#             action = agent.act(state)
#             next_state = take_action(state, actions[action])
#             r = reward(next_state)
#             done = next_state == goal_state
#             agent.remember(state, action, r, next_state, done)
#             state = next_state
#             total_reward += r

#             if done:
#                 print(f"Episode {e + 1}/{episodes}: Reward: {total_reward}, Steps: {time_step + 1}")
#                 break

#             agent.replay(batch_size)
#         agent.reduce_epsilon()
#         agent.update_target_model()

# # Run training
# train_dqn(episodes=500, batch_size=32)
