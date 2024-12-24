>[!abstract] Learning outcomes
>- Apply policy gradients and actor critic methods to solve small-scale MDP problems manually and program policy gradients and actor critic algorithms to solve medium-scale MDP problems automatically.
>- Compare and contrast policy-based reinforcement learning with value-based reinforcement learning.

## Overview
In policy gradient methods, we approximate the policy from the rewards and actions received in our episodes, similar to Q-learning, with two properties:
- The policy is represented using some function that is **differentiable** with respect to its parameters. For a non-differentiable policy, we cannot calculate the gradient.
- Typically, we want the policy to be **stochastic**. Recall from the section on [[Markov Decision Processes#Policies|policies]] that a stochastic policy specifies a **probability distribution** over actions, defining the probability with which each action should be chosen.
The goal of a policy gradient is to approximate the optimal policy $\pi_\theta(s,a)$ via gradient ascent on the expected return. Gradient ascent will find the best parameters θ for the particular MDP.
## Policy improvement using gradient ascent
The goal of gradient ascent is to find weights of a policy function that maximises the expected return. This is done iteratively by calculating the gradient from some data and updating the weights of the policy.
The expected value of a policy $\pi_\theta$ with parameters $\theta$ is defined as:
$$J(\theta)=V^{\pi_\theta}(s_0)$$
where $V^{\pi_\theta}$ is the policy evaluation using the policy $\pi_\theta$ and $s_0$ is the initial state. These search for a local maximum in $J(\theta)$ by **ascending the gradient** of the policy with respect to the parameters $\theta$.
>[!info] Definition - Policy gradient
>Given a policy objective $J(\theta)$, the **policy gradient** of $J$ with respect to $\theta$, $\nabla_\theta J(\theta)$ is defined as:
>$$\begin{split}
\nabla_{\theta}J(\theta) = \begin{pmatrix} \dfrac{\partial J(\theta)}{\partial \theta_1} \\ \vdots \\ \dfrac{\partial J(\theta)}{\partial \theta_n} \end{pmatrix}
\end{split}$$
where $\dfrac{\partial J(\theta)}{\partial\theta_i}$ is the partial derivative of $J$ with respective to $\theta_i$.

The gradient and update the weights to follow the gradient towards the optimal $J(\theta)$:
$$\theta \leftarrow \theta + \alpha \nabla J(\theta)$$
where $\alpha$ is a learning rate parameter that dictates how big the step in the direction of the gradient should be. According to **policy gradient theorem**, for any differentiable policy $\pi_\theta$, state $s$, and action $a$:
$$\nabla J(\theta) = \mathbb{E}\left[\nabla\ \textrm{ln} \pi_{\theta}(s, a) Q(s,a)\right]$$
## Reinforce
The REINFORCE algorithm is one algorithm for policy gradients. We cannot calculate the gradient optimally because this is too computationally expensive – we would need to solve for all possible trajectories in our model. In REINFORCE, we sample trajectories, similar to the sampling process in [[Temporal difference reinforcement learning#Monte-Carlo reinforcement learning|Monte-Carlo reinforcement learning]].
![[reinforce_algorithm.png]]The figure gives an abstract overview of REINFORCE. The algorithm iteratively generates new actions by sampling from its policy and executes these. Once an episode terminates, a list of the rewards and states is used to update the policy, showing that the policy is only updated at the end of each episode.
>[!todo] Algorithm 1 - REINFORCE
>**Input**: A differentiable policy $\pi_\theta(s,a)$, an MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
>**Output**: Policy $\pi_\theta(s,a)$
>Initialize parameters $\theta$
>**repeat**
>$\qquad$Generate episode $(s_0,a_0,r_1,...,s_{T-1},a_{T-1},r_T)$ by following $\pi_\theta$
>$\qquad$**for each** $(s_t,a_t)$ in the episode
>$\qquad\qquad G\leftarrow\sum_{k=t+1}^T\gamma^{k-t-1}r_k$
>$\qquad\qquad \theta\leftarrow\theta+\alpha\gamma^tG\nabla\ln\pi_\theta(s,a)$
>**until** $\pi_\theta$ converges

REINFORCE generates an entire episode using Monte-Carlo simulation by following the policy so far by sampling actions using the stochastic policy $\pi_\theta$. It samples actions because on their probabilities; that is, $a \sim \pi_{\theta}(s,a)$. This is an *on policy* approach.
## Implementation: Logistic regression-based REINFORCE
```python
import random
from itertools import count

from model_free_learner import ModelFreeLearner

class REINFORCE(ModelFreeLearner):
    def __init__(self, mdp, policy) -> None:
        super().__init__()
        self.mdp = mdp
        self.policy = policy

    """ Generate and store an entire episode trajectory to use to update the policy """
    def execute(self, episodes=100, max_episode_length=float('inf')):
        total_steps = 0
        random_steps = 50
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            episode_reward = 0.0
            for step in count():
                action = self.policy.select_action(state, self.mdp.get_actions(state))
                (next_state, reward, done) = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_reward += reward * (self.mdp.discount_factor ** step)
                total_steps += 1

                if done or step == max_episode_length:
                    break

            deltas = self.calculate_deltas(rewards)

            self.policy.update(states, actions, deltas)
            episode_rewards.append(episode_reward)

        return episode_rewards

    def calculate_deltas(self, rewards):
        G = []
        G_t = 0

        for r in reversed(rewards):
            G_t = r + self.mdp.get_discount_factor() * G_t 
            G.insert(0, G_t)

        return G
```
The policy inherits from `StochasticPolicy`, which means that the policy is a [[Markov Decision Processes#Deterministic vs. stochastic policies|stochastic policy]] $\pi_\theta(s,a)$ that returns the probability of action a being executed in state $s$. The `select_action` is stochastic, as can be seen below – it selects between the two actions using the policies probability distribution.
```python
import math
import random

from policy import StochasticPolicy


""" A two-action policy implemented using logistic regression from first principles """
class LogisticRegressionPolicy(StochasticPolicy):
    """ Create a new policy, with given parameters theta (randomly if theta is None)"""
    def __init__(self, actions, num_params, alpha=0.1, theta=None):
        assert len(actions) == 2

        self.actions = actions
        self.alpha = alpha

        if theta is None:
            theta = [0.0 for _ in range(num_params)]
        self.theta = theta

    """ Select one of the two actions using the logistic function for the given state """
    def select_action(self, state, actions):
        # Get the probability of selecting the first action
        probability = self.get_probability(state, self.actions[0])

        # With a probability of 'probability' take the first action
        if random.random() < probability:
            return self.actions[0]
        return self.actions[1]

    """ Update our policy parameters according using the gradient descent formula:
          theta <- theta + alpha * G * nabla J(theta), 
          where G is the future discounted reward
    """
    def update(self, states, actions, deltas):
        for t in range(len(states)):
            gradient_log_pi = self.gradient_log_pi(states[t], actions[t])
            # Update each parameter
            for i in range(len(self.theta)):
                self.theta[i] += self.alpha * deltas[t] * gradient_log_pi[i]

    """ Get the probability of applying an action in a state """
    def get_probability(self, state, action):
        # Calculate y as the linearly weight product of the 
        # policy parameters (theta) and the state
        y = self.dot_product(state, self.theta)

        # Pass y through the logistic regression function to convert it to a probability
        probability = self.logistic_function(y)

        if action == self.actions[0]:
            return probability
        return 1 - probability

    """ Computes the gradient of the log of the policy (pi),
    which is needed to get the gradient of the objective (J).
    Because the policy is a logistic regression, using the policy parameters (theta).
        pi(actions[0] | state)= 1 / (1 + e^(-theta * state))
        pi(actions[1] | state) = 1 / (1 + e^(theta * state))
    When we apply a logarithmic transformation and take the gradient we end up with:
        grad_log_pi(left | state) = state - state * pi(left | state)
        grad_log_pi(right | state) = -state * pi(left | state)
    """
    def gradient_log_pi(self, state, action):
        y = self.dot_product(state, self.theta)
        if action == self.actions[0]:
            return [s_i - s_i * self.logistic_function(y) for s_i in state]
        return [-s_i * self.logistic_function(y) for s_i in state]

    """ Standard logistic function """
    @staticmethod
    def logistic_function(y):
        return 1 / (1 + math.exp(-y))

    """ Compute the dot product between two vectors """
    @staticmethod
    def dot_product(vec1, vec2):
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])
```
## Implementation: Deep REINFORCE
The policy uses a three layer network with the following:
- The first layer takes the state vector, so the input features are the features of the state.
- We have a hidden layer, with a default number of 64 hidden dimensions, but this is parameterized by the variable `hidden_dim` in the `__init__` constructor.
- The third and final layer is the output layer, which returns a categorical distribution with a dimensional the same size as the action space, so that each action is associated with a probability of being selected.
- We use a non-linear ReLU (rectified linear unit) between layers.
```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

from policy import StochasticPolicy


class DeepNeuralNetworkPolicy(StochasticPolicy):
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, state_space, action_space, hidden_dim=128, alpha=0.001, stochastic=True):
        self.state_space = state_space
        self.action_space = action_space

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # Whether to select an action stochastically or deterministically
        self.stochastic = stochastic

    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.policy_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    """ Select an action using a forward pass through the network """

    def select_action(self, state, actions):
        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32)
        action_logits = self.policy_network(state)

        # Mark out the actions that are unavailable
        mask = torch.full_like(action_logits, float('-inf'))
        mask[actions] = 0
        masked_logits = action_logits + mask

        action_distribution = Categorical(logits=masked_logits)
        if self.stochastic:
            # Sample an action according to the probability distribution
            action = action_distribution.sample()
        else:
            # Choose the action with the highest probability
            action_probabilities = torch.softmax(masked_logits, dim=-1)
            action = torch.argmax(action_probabilities)
        return action.item()

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()

        # Convert from a tensor encoding back to the action space
        return probabilities[action]

    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions)

        action_log_probs = self.evaluate_actions(states, actions)

        # Construct a loss function, using negative because we want to descend,
        # not ascend the gradient
        loss = -(action_log_probs * deltas).sum()

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

        # Take a gradient descent step
        self.optimiser.step()
        return loss

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    @classmethod
    def load(cls, state_space, action_space, filename):
        policy = cls(state_space, action_space)
        policy.policy_network.load_state_dict(torch.load(filename))
        return policy
```
## Advantages and disadvantages of policy gradients (compared to value-based techniques)
### Advantages
- **High-dimensional problems**: The major advantage of policy-gradient approaches compared to value-based techniques like Q-learning and SARSA is that they can handle high-dimensional action and state spaces, including actions and states that are continuous. This is because we do not have to iterate over all actions using $\arg\max_{a\in A}(s)$ as we do in value-based approaches. For continuous problems, $\arg\max_{a\in A}(s)$ is not possible to calculate, while for a high number of actions, the computational complexity is dependent on the number of actions
### Disadvantages
- **Sample inefficiency**: Since the policy gradients algorithm takes an entire episode to do the update, it is difficult to determine which of the state-action pairs are those that effect the value $G$ (the episode reward), and therefore which to sample.
- **Loss of explanation**: Model-free reinforcement learning is a particularly challenging case to understand and explain why a policy is making a decision. This is largely due to the model-free property: there are no action definitions that can used as these are unknown. However, policy gradients are particularly difficult because the values of states are unknown. With value-based approaches, knowing $V$ or $Q$ provides some insight into why actions are chosen by a policy.
## Takeaways
>[!success] Takeaways
>- **policy gradient methods** such as REINFORCE directly learn a policy instead of first learning a value function or Q-function.
>- Using trajectories, the parameters for a policy are updated by **following the gradient upwards** – the same as gradient descent but in the opposite direction.
>- Unlike policy iteration, policy gradient approaches are **model free**.

| Previous             |                     Next |
| :------------------- | -----------------------: |
| [[Policy iteration]] | [[Actor-critic methods]] |
