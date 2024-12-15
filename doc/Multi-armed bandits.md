>[!abstract] Learning outcomes
>- Select and apply multi-armed bandit algorithms for a given problem
>- Compare and contrast the strengths and weaknesses of different multi-armed bandit algorithms

## Overview
**Multi-armed bandit** techniques are not techniques for solving MDPs, but they are used throughout a lot of reinforcement learning techniques that do solve MDPs.
> [!info] Definition - Multi-armed bandit
> A **multi-armed bandit** (a.k.a an $N$-*armed bandit*) is defined by a set of random variables $X_{i,k}$, where:
> - $1\leq i\leq N$, such that $i$ is the **arm** of the bandit; and
> - $k$ the index of the **play** of arm $i$
> 
> Successive plays $X_{i,1},X_{j,2},X_{k,3}...$ are assumed to be independently distributed, but we do not know the probability distributions of the random variables.
> The idea is that a gambler iteratively plays rounds, observing the reward from the arm after each round, and can adjust their strategy each time. The aim is to maximize the sum of the rewards collected over all rounds. 
> Multi-arm bandit strategies aim to learn a **policy** $\pi(k)$, where $k$ is the play.

Given that we do not know the probability distributions, a simple strategy is simply to select the arm given a uniform distribution; that is, select each arm with the same probability. This is just uniform sampling.
Then, the Q-value for an action a can be estimated using the following formula:
$$Q(a)=\dfrac{1}{N(a)}\sum_{i=1}^t{X_{a,i}}$$
where $t$ is the number of rounds so far, $N(a)$ is the number of times $a$ selected in the previous rounds, and $X_{a,i}$ is the **reward** obtained in the $i$-th round for playing arm $a$.
The idea here is that for a multi-armed bandit problem, we explore the options uniformly for some time, and then once we are confident we have enough samples (when the changes to the values of $Q(a)$ start to stabilize), we start selecting $\arg\max_aQ(a)$. This is known as the *$\epsilon$-first* strategy, where the parameter $\epsilon$, determines how many rounds to select random actions before moving to the greedy action.
## Exploration vs. Exploitation
What we want is to play only the good actions; so just keep playing the actions that have given us the best reward so far. However, at first, we do not have information to tell us what the best actions are. Thus, we want strategies that **exploit** what we think are the best actions so far, but still **explore** other actions.
>[!info] Definition - Regret
>Given a policy $\pi$ and $t$ number of arm pulls, **regret** is defined as:
>$$\mathcal{R}(\pi,t)=t\cdot\max_a{Q^*(a)}-\mathbb{E}\left[\sum_{k=1}^t{X_{\pi(k),k}}\right]$$
>where $Q^*(a)$ is actual average return of playing arm $a$. We do not know $Q^*(a)$ of course - otherwise we could simply play $\arg\max_a{Q^*(a)}$ each round.

Informally: If we follow policy $\pi$ by playing arm $\pi(k)$ in round each round $k$, our regret over the $t$ pulls is the _best possible cumulative reward_ minus the _expected reward of playing using policy $π$_. So, regret is the **expected loss** from not taking the best action. If I take always action $\arg\max_aQ^*(a)$ (the best action), my regret is 0.
The aim of a multi-armed bandit strategy to learn a policy that minimizes the total regret.
A **zero-regret** strategy is a strategy whose average regret each round approaches zero as the number of rounds approached infinity. So, this means that a zero-regret strategy will converge to an optimal strategy given enough rounds.
## Implementation
The implementation for each strategy we discuss inherits from the class `MultiArmedBandit`:
```python
class MultiArmedBandit():
    """ Select an action for this state given from a list given a Q-function """
    def select(self, state, actions, qfunction):
        abstract

    """ Reset a multi-armed bandit to its initial configuration """
    def reset(self):
        self.__init__()
```

Each strategy must implement the `select` method, which takes the list of available actions and their Q-values. The `reset` method resets to the bandit to its initial configuration, and is used for demonstration purposes later.
## $\epsilon$-greedy strategy
The $\epsilon$-greedy strategy is a simple and effective way of balancing exploration and exploitation. In this algorithm, the parameter $\epsilon\in[0,1]$  controls how much we explore and how much we exploit.
Each time we need to choose an action, we do the following:
- With probability $1−\epsilon$ we choose the arm with the maximum $Q$ value: $\arg\max_aQ(a)$ (**exploit**). If there is a tie between multiple actions with the largest $Q$-value, break the tie randomly.
- With probability $\epsilon$ we choose a random arm with uniform probability (**explore**).   
The best value for $\epsilon$ depends on the particular problem, but typically, values around 0.05-0.2 work well as they exploit what they have learnt, while still exploring.
### Implementation
The implementation for epsilon greedy uses `random()` to select a random number between 0 and 1. If that number is less than epsilon, an action is randomly selected. If the number is greater than or equal to epsilon, it finds the actions with the maximum Q value, breaking ties randomly:

```python
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def reset(self):
        pass

    def select(self, state, actions, qfunction):
        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(actions)
        arg_max_q = qfunction.get_argmax_q(state, actions)
        return arg_max_q
```
## $\epsilon$-decreasing strategy
**$\epsilon$-decreasing** follows a similar idea to epsilon greedy, however, it recognizes that initially, we have very little feedback so exploiting is not a good strategy to being with: we need to explore first. Then, as we gather more data, we should exploit more.
The $\epsilon$-decreasing strategy does this by taking the basic epsilon greedy strategy and introducing another parameter $\alpha\in[0,1]$, which is used to decrease $ϵ$ over time. For this reason, $\alpha$ is called the **decay**.
The selection mechanism is the same as epsilon greedy:
- explore with probability $\epsilon$; and 
- exploit with probability $1−\epsilon$.
However, after each selection, we decay $\epsilon$ using $\epsilon\leftarrow\epsilon\times\alpha$. We start initially with a higher value of $\epsilon$, such as $\epsilon=1.0$, to ensure that we explore a lot, and it will slowly decay to a low number such that we explore less and less as we gather more feedback.
### Implementation
The following implementation for the ϵ-decreasing strategy uses the ϵ-greedy strategy, just decreasing the epsilon value each step:
```python
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

class EpsilonDecreasing(MultiArmedBandit):
    def __init__(self, epsilon=1.0, alpha=0.999, lower_bound=0.1):
        self.epsilon_greedy_bandit = EpsilonGreedy(epsilon)
        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.lower_bound = lower_bound

    def reset(self):
        self.epsilon_greedy_bandit = EpsilonGreedy(self.initial_epsilon)

    def select(self, state, actions, qfunction):
        result = self.epsilon_greedy_bandit.select(state, actions, qfunction)
        self.epsilon_greedy_bandit.epsilon = max(
            self.epsilon_greedy_bandit.epsilon * self.alpha, self.lower_bound
        )
        return result
```
In this implementation, we have a minimum value, `lower_bound`, such that $\epsilon$ stop decaying. As $\epsilon$ approaches 0, the bandit stops exploring and just exploits. During learning, we do not want this, so we add a lower bound.
## Softmax strategy
**Softmax** is a probability matching strategy, which means that the probability of each action being chosen is dependent on its $Q$-value so far. Formally, softmax chooses an action because on the Boltzmann distribution for that action:
$$\dfrac{e^{Q(a)/\tau}}{\sum_{b=1}^Ne^{Q(b)/\tau}}$$
where $N$ is the number of arms, and $\tau>0$ is the **temperature**, which dictates how much of an influence the past data has on the decision. A higher value of $\tau$ would mean that the probability of selecting each action is close to each other (as $\tau$ approaches infinity, softmax approaches a uniform strategy), while a lower value of $\tau$ would imply that the probabilities are closer to their $Q$ values. When $\tau=1$, the probabilities are just $e^{Q(a)}$, and as $\tau$ approaches 0, softmax approaches a greedy strategy.
As with the $\epsilon$-decreasing strategy, we can add a decay parameter α that allows the value of $\tau$ to decay until it reaches 1. This encourages exploration in earlier phases, and exploration less as we gather more feedback.
### Implementation
The following implementation of the softmax strategy uses `random()` to generate a random number between 0 and 1, and divides this space 0-1 among the set of actions based on the value of $e^{Q(a)}/\tau$:
```python
import math
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class Softmax(MultiArmedBandit):
    def __init__(self, tau=1.0):
        self.tau = tau

    def reset(self):
        pass

    def select(self, state, actions, qfunction):

        # calculate the denominator for the softmax strategy
        total = 0.0
        for action in actions:
            total += math.exp(qfunction.get_q_value(state, action) / self.tau)

        rand = random.random()
        cumulative_probability = 0.0
        result = None
        for action in actions:
            probability = (
                math.exp(qfunction.get_q_value(state, action) / self.tau) / total
            )
            if cumulative_probability <= rand <= cumulative_probability + probability:
                result = action
            cumulative_probability += probability

        return result
```
## Upper Confidence Bounds (UCB1)
A highly effective multi-armed bandit strategy is the **Upper Confidence Bounds** (UCB1) strategy. Using the UCB1 strategy, we select the next action using the following:
$$\arg\max_a{\left(Q(a)+\sqrt{\dfrac{2\ln t}{N(a)}}\right)}$$
where $t$ is the number of rounds so far, and $N(a)$ is the number of times times a has been chosen in all previous rounds. The term inside the square root is undefined if $N(a)=0$. The avoid this, the typical strategy is to spend the first $N$ rounds to select each of the $N$ bandits once. This is a simple exploration strategy to ensure all arms are sampled at least once, but this problem could be handled in different ways, such as assigning the value $\infty$ when $N(a)=0$.
- The left–hand side encourages exploitation: the $Q$-value is high for actions that have had a high reward.
- The right–hand side encourages exploration: it is high for actions that have been explored less – that is, when $N(a)$ relative to other actions. When $t$ is small (not many pull so far), all actions will have a high exploration value. As t increases, if some actions have low $N(a)$, then the expression $\sqrt{\dfrac{2\ln t}{N(a)}}$ is large compared to actions with higher $N(a)$.
Together, adding these two expressions helps to balance exploration and exploitation. Note that the UCB formula is not parameterized - that is, there is no parameter giving weight to the $Q(a)$ or the square root expression to balance exploration vs. exploitation.
We want to learn the $Q$-function, which gives us the average return on each action $a$, such that it approximates the real (unknown) $Q$-function, which we will call $Q^∗$. At each round, we select the action a that maximizes the expression inside the brackets. If arm $a$ is optimal, then we want the following to hold for all actions $b\neq a$:
$$Q(b)+\sqrt{\dfrac{2\ln ⁡t}{N(b)}}\leq Q^*(a)$$
If this holds, we have some confidence that $Q(a)$ is optimal. If $N(b)$ is low for some actions, we do not have this confidence. So, the expression $\sqrt{\dfrac{2\ln t}{N(a)}}$ is the **confidence interval** of our estimates for $Q(a)$, much like a confidence interval around an estimation of a population mean in statistic.
If by chance the above expression does NOT hold for the optimal action $a$, then $a$ is not chosen, but it should be. We want this to occur only with probability $\dfrac{1}{N}$ to minimize pseudo-regret. This leads us to $2\ln t⁡$ in the expression: if there has been relatively few pulls overall so far, then the confidence intervals will be similar for all arms. As more pulls are done, this increases, but only at a logarithmic rate so that exploration grows more slowly as $t$ increases.
### Implementation
```python
import math
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class UpperConfidenceBounds(MultiArmedBandit):
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, state, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        for action in actions:
            value = qfunction.get_q_value(state, action) + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)
        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1
        return result
```
## Comparison
![[multi-armed bandit.png]]
UCB1, on average, obtains the highest reward for the simulation. If we extend the simulation episodes to be longer, we would see that eventually epsilon decreasing would start to achieve similar rewards to UCB1, but it takes longer to converge to this.
## Takeaways
>[!success] Takeaways
>- Multi-armed bandits are problems in where we must make a selection from a set of options, but we do not know the probability of success (or the expected return) of each options
>- Several techniques can be used to solve these, including: $\epsilon$-greedy, $\epsilon$-decreasing, softmax, and UCB1 strategy.
>- In a simple experiment, we found that UCB1 was the fastest learner and recovered from shift quickly. However, this is just one experiment – other domains will have different properties.

| Previous            |                                           Next |
| :------------------ | ---------------------------------------------: |
| [[Value Iteration]] | [[Temporal difference reinforcement learning]] |
