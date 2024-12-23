>[!abstract] Learning outcomes
>- Manually apply linear Q-function approximation to solve small-scale MDP problems given some known features.
>- Select suitable features and design & implement Q-function approximation for model-free reinforcement learning techniques to solve medium-scale MDP problems automatically.
>- Argue the strengths and weaknesses of function approximation approaches.
>- Compare and contrast linear Q-learning with deep Q-learning.

## Overview
Using a Q-table has two limitations:
- It requires visiting every reachable state many times and apply every action many times to get a good estimate of $Q(s,a)$. Thus, if a state $s$ is never visited, there is no estimate of $Q(s,a)$.
- It requires us to maintain a table of size $|A|\times|S|$, which is prohibitively large for any non-trivial problem.
$\to$ Use machine learning to approximate Q-functions. In particular, we will look at **linear function approximation** and approximation using **deep learning** (deep Q-learning). Instead of calculating an exact Q-function, we approximate it using simple methods that both eliminate the need for a large Q-table (therefore the methods scale better), and also allowing use to provide reasonable estimates of $Q(s,a)$ _even if we have not applied action a in state s previously_.
## Linear Q-learning (Linear Function Approximation)
The key idea is to **approximate** the Q-function using a linear combination of **features** and their **weights**. Instead of recording everything in detail, we think about what is most important to know, and model that. The overall process is:
- for the states, consider what are the features that determine its representation;
- during learning, perform updates based on the **weights of features** instead of states; and
- estimate $Q(s,a)$ by summing the features and their weights.
### Linear Q-function representation
In linear Q-learning, we store features and weights, not states. We need to learn how important each feature (**weight**) for each action. To represent this, we have two actions:
- A **feature vector**, $f(s,a)$, which is a vector of $n\cdot|A|$ different functions, where $n$ is the number of state features and $|A|$ is the number of actions.Each function extract the value of a feature for state-action pair $(s,a)$. Note that $f_i(s,a)$ extracts the $i$th feature from the state-action pair $(s,a)$:
$$\begin{split}f(s,a) = \begin{pmatrix} 
        f_1(s,a) \\
        f_2(s,a) \\
        \ldots\\
        f_{n \times |A|}(s,a) \\
        \end{pmatrix}\end{split}$$
- A **weight vector** $w$ of size $n\times|A|$: one weight for each feature-action pair; $w_i^a$ defines the weight of a feature $i$ for action $a$.
### Defining State-Action features
Often it is easier to just define features for states, rather than state-action pairs. The features are just a vector of $n$ function of the form $f_i(s)$. However, for most applications, the weight of a features are just a vector of $n$ functions of the form $f_i(s)$.
It is straightforward to construct $n\times|A|$ state-pair features from just $n$ state features:
$$\begin{split}
f_{i,k}(s,a) = \Bigg \{
\begin{array}{ll}
 f_i(s) & \text{if } a=a_k\\
 0      & \text{otherwise}
 ~~~ 1 \leq i \leq n, 1 \leq k \leq |A|
 \end{array}
\end{split}$$
This effective results in $|A|$ different weight vectors:
$$\begin{split}
 f(s,a_1) = \begin{pmatrix} 
f_{1,a_1}(s,a) \\
f_{2,a_1}(s,a) \\
0\\
0\\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_2) = \begin{pmatrix} 
0\\
0\\
f_{1,a_2}(s,a) \\
f_{2,a_2}(s,a) \\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_3) = \begin{pmatrix} 
0\\
0\\
0\\
0\\
f_{1,a_3}(s,a) \\
f_{2,a_3}(s,a) \\
\ldots
\end{pmatrix}~~\ldots
\end{split}$$
### Q-values from linear Q-functions
Give a feature vector $f$ and a weight vector $w$, the Q-value of a state is a simple linear combination of features and weights:
$$\begin{split}
\begin{array}{lll}
  Q(s,a) & = & f_1(s,a) \cdot w^a_1 + f_2(s,a)\cdot w^a_2 + \ldots  + f_{n}(s,a) \cdot w^a_n\\
         & = & \sum_{i=0}^{n} f_i(s,a) w^a_i
\end{array}
\end{split}$$
In practice, we also multiple the feature vector for weights $w_n^b$ for all actions $b\neq a$, but as the feature values will be 0, we know that it does not influence the result.
### Linear Q-function update
To use approximate Q-functions in reinforcement learning, there are two steps we need to change from the standard algorithms: (1) initialization; and (2) update.
- For initialization, initialize all weights to 0. Alternatively, you can try Q-function initialization and assign weights that you think will be "good" weights.
- For update, we now need to update the weights instead of the Q-table values. The update rule is now:
$$w^a_i \leftarrow w^a_i + \alpha \cdot \delta \cdot \ f_i(s,a)$$
where $\delta$ depends on which algorithm (Q-learning, SARSA).
### Implementation
```python
from qfunction import QFunction

class LinearQFunction(QFunction):
    def __init__(self, features, alpha=0.1, weights=None, default_q_value=0.0):
        self.features = features
        self.alpha = alpha
        if weights == None:
            self.weights = [
                default_q_value
                for _ in range(0, features.num_actions())
                for _ in range(0, features.num_features())
            ]

    def update(self, state, action, delta):
        # update the weights
        feature_values = self.features.extract(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (self.alpha * delta * feature_values[i])

    def get_q_value(self, state, action):
        q_value = 0.0
        feature_values = self.features.extract(state, action)
        for i in range(len(feature_values)):
            q_value += feature_values[i] * self.weights[i]
        return q_value
```
A linear Q-function is initialized with either some given weights or with a default weight for all weights. The `update` updates each weight by adding $\delta\cdot f_i(s,a)$.
### Challenges and tips
he key challenge in linear function approximation for Q-learning is the feature engineering: selecting features that are meaningful and helpful in learning a good Q function.
## Deep Q-learning
The latest hype in reinforcement learning is all about the use of deep neural networks to approximate value and Q-functions.
### Deep Q-function representation
In deep Q-learning, Q-functions are represented using deep neural networks. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a;\theta)$, so takes the parameters as an argument.
This has the advantage (over linear Q-function approximation) that feature engineering is not required, the "features" will be learnt as part of the hidden layers of the neural network. A further advantage is that states can be non-structured (or less structured), rather than using a factored state representation.
### Deep Q-function update
The update rule for deep Q-learning looks similar to that of updating a linear Q-function. The deep reinforcement learning TD update is:
$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_{\theta} Q(s,a; \theta)$$
where $\nabla_\theta Q(s,a;\theta)$  is the **gradient** of the Q-function.
### Implementation
Deep Q-learning is identical to tabular or linear Q-learning, except that we use a deep neural network to represent the Q-function instead of a Q-table or a linear equation.
Using [Pytorch](https://pytorch.org/), we create a sequential neural network with the following:
- The first layer takes the state vector, so the input features are the features of the state.
- We have a hidden layer, with a default number of 64 hidden dimensions, but this is parameterized by the variable `hidden_dim` in the `__init__` constructor.
- The third and final layer is the output layer, whose dimension is the same as the action space, so that each action has a Q-value associated with it.
- Using a non-linear ReLU between layers.
The input, hidden, and output layers are all `Linear` layers. We also need to implement the `get_q_value` method to pass through the network to get our Q-values.
```python
import random
import torch
import torch.nn as nn
from torch.optim import Adam

from qfunction import QFunction


class DeepQFunction(QFunction):
    """A neural network to represent the Q-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(self, state_space, action_space, hidden_dim=128, alpha=0.001):

        # Create a sequential neural network to represent the Q function
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_space),
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=alpha, amsgrad=True)

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.q_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    def update(self, state, action, delta):
        return self.batch_update([state], [action], [delta])

    def batch_update(self, experiences):
        (states, actions, deltas, dones) = zip(*experiences)
        return self.batch_update(states, actions, deltas)

    def batch_update(self, states, actions, deltas):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        q_values = (
            self.q_network(states_tensor)
            .gather(dim=1, index=actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Construct the target values
        targets = [value + delta for value, delta in zip(q_values.tolist(), deltas)]
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)

        loss = nn.functional.smooth_l1_loss(
            q_values,
            targets_tensor,
        ).sum()

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimiser.step()
        return loss

    def get_q_values(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        with torch.no_grad():
            q_values = self.q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            )
        return q_values.squeeze(1).tolist()

    def get_max_q_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            max_q_values = self.q_network(states_tensor).max(1).values
        return max_q_values.tolist()

    def get_q_value(self, state, action):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        q_value = q_values[action].item()

        return q_value

    def get_max_pair(self, state, actions):
        # Convert the state into a tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        max_q = float("-inf")
        max_actions = []
        for action in actions:
            q_value = q_values[action].item()
            if q_value > max_q:
                max_actions = [action]
                max_q = q_value
            elif q_value == max_q:
                max_actions += [action]

        arg_max_q = random.choice(max_actions)
        return (arg_max_q, max_q)

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.q_network.state_dict()
        policy_dict = policy_qfunction.q_network.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.q_network.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
```
### Advantages and disadvantages
**Advantages** of deep Q-function approximation (compared to linear Q-function approximation):
- **Feature selection**: We do not need to select features – the ‘features’ will be learnt as part of the hidden layers of the neural network.
- **Unstructured data**: The state s can be less structured, such as images or sequences of images (video).
**Disadvantages**:
- **Convergence**: There are no convergence guarantees.
- **Data hungry**: Deep neural networks are more data hungry because they need to learn features as well as "the Q-function", so compared to a linear approximation with good features, learning good Q-functions can be difficult. Large amounts of computation are often required.
Despite this, deep Q-learning works remarkably well in some areas, especially for tasks that require vision.
## Strengths and Limitations of Q-function Approximation
Approximating Q-functions using machine learning techniques such as linear functions or deep learning has advantages and disadvantages.
**Advantages:**
- Memory: More efficient representation compared with Q-tables because we only store weights/parameters for the Q-function, rather than the the $|A|\times|S|$ entries for a Q-table.
- Q-value propagation: we do not need to apply action a in state s to get a value for $Q(s,a)$ because the Q-function generalizes.
**Disadvantages:**
- The Q-function is now only an approximation of the real Q-function: states that share feature values will have the same Q-value according to the Q-function, but the actual Q-value according to the (unknown) optimal Q-function may be different.
## Takeaways
>[!success] Takeaways
>- We can scale reinforcement learning by **approximating Q-functions**, rather than storing complete Q-tables.
>- Using simple **linear methods** in which we select features and learn weights are effective and guarantee convergence.
>- **Deep Q-learning** offers an alternative in which we can learn a state representation, but requires more training data (more episodes) and has no convergence guarantees.


| Previous                    |               Next |
| :-------------------------- | -----------------: |
| [[Monte-Carlo Tree Search]] | [[Reward shaping]] |
