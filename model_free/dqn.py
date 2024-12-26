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
    def __init__(self, mdp:MDP, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)

        # Neural networks
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state_tensor = state_to_tensor(state, self.model.device).unsqueeze(0)
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
            state_tensor = state_to_tensor(state, self.model.device)
            next_state_tensor = state_to_tensor(next_state, self.model.device)
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
    def state_to_tensor(state, device):
        grid = np.zeros(grid_size)
        grid[state] = 1
        return torch.tensor(grid.flatten(), dtype=torch.float32).to(device)