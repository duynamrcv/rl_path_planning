import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from mdp_problem.mdp import MDP

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.fc(state)

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.fc(state)
    
class ActorCriticLearner():
    def __init__(self, mdp:MDP, gamma=0.99, learning_rate=1e-3):
        self.mdp = mdp
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states = self.mdp.get_states()
        self.actions = self.mdp.get_actions()

        self.actor = Actor(self.states[0] * self.states[1], len(self.actions)).to(self.device)
        self.critic = Critic(self.states[0] * self.states[1]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def state_to_tensor(self, state):
        grid = np.zeros(self.states)
        grid[state] = 1
        return torch.tensor(grid.flatten(), dtype=torch.float32).to(self.device)

    def learn(self, episodes=1000):
        total_rewards = []
        for episode in range(episodes):
            # Reset environment and model
            self.mdp.reset()
            state = self.mdp.get_current_state()
            total_reward = 0
            total_step = 0
            done = False
            while not done:
                # Convert state to tensor
                state_tensor = self.state_to_tensor(state).unsqueeze(0)

                # Select action
                action_probs = self.actor(state_tensor)
                action = np.random.choice(len(self.actions), p=action_probs.cpu().detach().numpy()[0])
                reward, done = self.mdp.step(self.actions[action])
                next_state = self.mdp.get_current_state()
                total_reward += reward

                # Convert next state to tensor
                next_state_tensor = self.state_to_tensor(next_state).unsqueeze(0)

                # Critic value estimation
                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)

                # Compute advantage and targets
                target = reward + (self.gamma * next_value.item() if next_state != self.mdp.get_goal_states() else 0)
                target_tensor = torch.tensor([[target]], dtype=torch.float, device=self.device)  # Ensure Float type
                advantage = target - value.item()

                # Update Actor
                actor_loss = -torch.log(action_probs[0, action]) * advantage
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update Critic
                critic_loss = nn.MSELoss()(value, target_tensor)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Transition to next state
                state = next_state

                total_step += 1
            
            if done:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {total_step + 1}")
            total_rewards.append(total_reward)
        return total_rewards
    
    def save_models(self, episode, directory="models"):
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.actor.state_dict(), f"{directory}/actor_model_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{directory}/critic_model_{episode}.pth")
        print(f"Models saved for episode {episode}")

    def execution(self, init, goal, episodes=1000):
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
        total_rewards = self.learn(episodes=episodes)
        id_sort = np.argsort(total_rewards)
        self.save_models(id_sort[-1])

        # Extract optimal path
        self.mdp.reset()
        state = self.mdp.get_current_state()
        done = False
        path = [state]
        while not done:
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            action_probs = self.actor(state_tensor)
            action = np.random.choice(len(self.actions), p=action_probs.cpu().detach().numpy()[0])
            _, done = self.mdp.step(self.actions[action])
            state = self.mdp.get_current_state()
            path.append(state)
        return path, total_rewards