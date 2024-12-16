> [!abstract]  Learning outcomes
> - Identify situations in which model-free reinforcement learning is a suitable solution for an MDP.
> - Explain how model-free planning differs from model-based planning
> - Apply temporal difference methods Q-learning and SARSA to solve small- and medium-scale MDP problems
> - Compare and contrast off-policy and on-policy reinforcement learning

## Model-based vs. model-free
Value iteration is a part of solutions known as **model-based** techniques. This means that we need to know the model; in particular, we have to access to $P_a(s'|s)$ and $r(s,a,s')$.
Q-learning and SARSA are **model-free** techniques. This means that we do not known $P_a(s'|s)$ and $r(s,a,s')$ of model.
Importantly, in model-free reinforcement learning, we do not try to learn $P_a(s'|s)$ or $r(s,a,s')$, we learn a value function or a policy directly.
There are something between model-based and model-free: simulation-based techniques. We have a model as a **simulator**, so we can **simulate** $P_a(s'|s)$ and $r(s,a,s')$  and learn a policy with a model-free technique, but we can not "see" $P_a(s'|s)$ and $r(s,a,s')$, so model-based techniques like value iteration are not possible.
## Intuition of model-free reinforcement learning
![[model-free.png]]
There are many different techniques for model-free reinforcement learning, all with the same basis as above figure gives an abstract illustration of this process.
- Execute many different **episodes** of the problem we want to solve in order to learn a **policy**.
- During each episode, we loop between executing actions and learning our policy.
- When our agent executes an action, we get a reward and we can see the new state that results from executing the action.
- From this, we **reinforce** our estimates of applying the previous action in the previous state.
- Then, we select a new action and execute it in the environment.
- We repeat this until either:
	- (1) run out of training time
	- (2) policy has converged to an optimal policy
	- (3) policy is 'good enough'
## Monte-Carlo reinforcement learning
 Monte-Carlo reinforcement learning is perhaps the simplest of reinforcement learning methods.The intuition is quite straightforward. Maintain a Q-function that records the value $Q(s,a)$ for every state-action pair:
 - (1) choose an action using a multi-armed bandit algorithm;
 - (2) apply that action and receive the reward;
 - (3) update $Q(s,a)$ based on that reward
It is called **Monte-Carlo reinforcement learning**.
>[!todo] Algorithm 1 - Monte-Carlo reinforcement learning
>**Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
>**Output**: Q-function $Q$
>Initialize $Q(s,a)\leftarrow0$ for all $s$ and $a$
>$N(s,a)\leftarrow0$ for all $s$ and $a$
>**repeat**
>$\qquad$ Generate an episode $(s_0,a_0,r_1,...,s_{T-1},a_{T-1},r_{T})$
>$\qquad G\leftarrow0$
>$\qquad t\leftarrow T-1$
>$\qquad$ **while** $t\geq0$ **do**
>$\qquad\qquad G\leftarrow r_{t+1}+\gamma G$
>$\qquad\qquad$**if** $s_t,a_t$ does not appear in $s_0,a_0,...,s_{t-1},a_{t-1}$**then**
>$\qquad\qquad\qquad Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\dfrac{1}{N(s_t,a_t)}[G-Q(s_t,a_t)]$
>$\qquad\qquad\qquad N(s_t,a_t)\leftarrow N(s_t,a_t)+1$
>$\qquad\qquad t\leftarrow t-1$
>**until** $Q$ converges

This algorithm generates an entire episode following some policy, such $\epsilon$-greedy, observing the reward $r_{t+1}$ at each step $t$. It then calculates the discounted future reward $G$ at each step. If $s_t$, $a_t$ occurs earlier in the episode, then we do not update $Q(s_t,a_t)$ as we will update it later in the loop. If it does not occur, we update $Q(s_t,a_t)$ as the cumulative average over all executions of $s_t$, $a_t$ over all episodes. In this algorithm $N(s_t,a_t)$ represents the number of times that $s_t$, $a_t$ have been evaluated over all episodes.
## Q-tables
Q-table is the simplest way to maintain a Q-function. It is a table with an entry for every $Q(s,a)$. Thus, like value functions in value iteration, they do not scale to large state-spaces.
## Temporal difference (TD) reinforcement learning
Monte-Carlo reinforcement learning is simple, but it has a number of problems. The most important is that it has high **variance**. Recall that the future discounted reward for an episode and use that to calculate the average reward for each state-action pair.  However, the term $\gamma G$ is often *not* a good estimation of the average future award. If we execute action $a$ in state $s$ many times throughout different episodes, we might find that the future trajectories we execute after that vary significantly because we are using Monte Carlo simulation. This means that it will take a long term to learn a good estimate of the true average reward is for that state-action pair.
**Temporal difference** (TD) methods alleviate this problem using **bootstrapping**. Much the same way that value iteration bootstraps by using the last iteration’s value function, in TD methods, instead of updating based on $G$ - the actual future discounted reward received in the episode – we update based on the actual immediate reward received plus an estimate of our future discounted reward.
![[temporal-difference.png]]
In TD method, update rules always follow a pattern:
$$Q(s,a) \leftarrow \underbrace{Q(s,a)}_\text{old value} + \overbrace{\alpha}^{\text{learning rate}} \cdot [\underbrace{\overbrace{r}^{\text{reward}} + \overbrace{\gamma}^{\text{discount factor}} \cdot V(s')}_{\text{TD target}} - \overbrace{Q(s,a)}^{\text{do not count extra } Q(s,a)}]$$
$V(s')$ is TD estimate of the  average future reward, and is the bootstrapped value of future discounted reward. The new information is weighted by a parameter $\alpha\in[0,1]$, which is the **learning rate**. A higher learning rate $\alpha$ will weight more recent information higher than older information, so will learn more quickly, but will make it more difficult to stabilize because it is strongly influenced by outliers.
The idea is that over time, the TD estimate will become be more stable than the actual rewards we receive in an episode, converging to the optimal value function $V(s')$ defined by the Bellman equation, which leads to Q(s,a) converging more quickly.
Why is there the expression $-Q(s,a)$ inside the square brackets? This is because the old value is weighted $(1−\alpha)\cdot Q(s,a)$. We can expand this to $Q(s,a)−\alpha\cdot Q(s,a)$, and can then move the latter $−Q(s,a)$ inside the square brackets where the learning rate α is applied.
Two TD methods that differ in the way that they estimate the future reward $V(s')$:
- **Q-learning** is **off-policy** approach; and
- **SARSA** is **on-policy** approach
## Q-Learning: Off-policy temporal-difference learning
Q-learning is a foundation method for reinforcement learning. It is TD method that estimates the future reward $V(s')$ using Q-function, assuming that from state $s'$, the best action (according to $Q$) will be executed at each time.
Q-learning algorithm uses $\max_{a'}{Q(s',a')}$ as the estimation of $V(s')$, that is, it estimates $Q$ to estimate the value of the best action from $s'$.
>[!todo] Algorithm 2 - Q-learning
> **Input**: MDP $M=\left\langle S,s_0,A,P_a(s'a|s),r(s,a,s')\right\rangle$
> **Output**: Q-function $Q$
> Initialize $Q(s,a)\leftarrow0$ for all $s$ and $a$
> **repeat**
> $\qquad s\leftarrow$ the first state in episode $e$
> $\qquad$ **repeat** each step in episode $e$
> $\qquad\qquad$Select action $a$ to apply in $s$
> $\qquad\qquad$Execute action $a$ in state $s$
> $\qquad\qquad$Observe reward $r$ and new state $s'$
> $\qquad\qquad \delta\leftarrow r+\gamma\max_{a'}Q(s',a')-Q(s,a)$
> $\qquad\qquad Q(s,a)\leftarrow Q(s,a)+\alpha\delta$
> $\qquad\qquad s\leftarrow s'$
> $\qquad s$  is the last state of episode $e$ (a terminal state)
> **until** $Q$ converges

### Updating the Q-function
Updating the Q-function:
$$\delta \leftarrow [\underbrace{\overbrace{r}^{\text{reward}} + \overbrace{\gamma}^{\text{discount factor}} \cdot \overbrace{\max_{a'} Q(s',a')}^{V(s') \text{ estimate}}}_{\text{TD target}}  \overbrace{- Q(s,a)}^{\text{do not count extra } Q(s,a)}]$$
$$Q(s,a) \leftarrow 
\underbrace{Q(s,a)}_\text{old value} + \overbrace{\alpha}^{\text{learning rate}} \cdot \underbrace{\delta}_{\text{delta value}}$$
At each step, $Q(s,a)$ is update by taking the old value of $Q(s,a)$ and adding this to the new information. The estimate from the new observations is given by $\delta\leftarrow r+\gamma\max_{a'}Q(s',a')$, where $\delta$ is the difference between the previous estimate and the most recent observation, $r$ is the reward that was received by executing action $a$ in state $s$, and $r+\gamma\max_{a'}Q(s',a')$ is the **temporal difference** target. What this says is that the estimate of $Q(s,a)$ based on the new information is the reward $r$, plus the estimated discounted future reward from being in state $s'$. The definition of $\delta$ is the update similar to that of the Bellman equation. We do not know $P_a(s'|s)$, so we cannot calculate the Bellman update directly, but we can estimate the value using $r$ and the temporal difference target.
>[!note] Note
>Note that we estimate the future value using $\max_{a'}Q(s',a')$, which means it _ignores_ the actual next action that will be executed, and instead updates based on the **estimated best action** for the update. This is known as **off policy** learning — more on this later.

### Policy extraction using Q-functions
Using [[Markov Decision Processes#Policy extraction|policy extraction]] exactly doing for value iteration, except extracting from the Q-function instead of the value function:
$$\pi(s) = \text{argmax}_{a \in A(s)} Q(s,a)$$
This selects the action with the maximum Q-value. Given an optimal Q-function (for the MDP), this results in optimal behavior.
### Implementation
To implement Q-learning, we first implement an abstract superclass `ModelFreeLearner` that defines the interface for any model-free learning algorithm:
```python
class ModelFreeLearner:
    def execute(self, episodes=2000):
        abstract
```
Next, we implement a second abstract super-class `TemporalDifferenceLearner`, which contains most of the code we need:
```python
from itertools import count
from model_free_learner import ModelFreeLearner

class TemporalDifferenceLearner(ModelFreeLearner):
    def __init__(self, mdp, bandit, qfunction):
        self.mdp = mdp
        self.bandit = bandit
        self.qfunction = qfunction

    def execute(self, episodes=2000, max_episode_length=float("inf")):
        episode_rewards = []
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            episode_reward = 0.0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                delta = self.get_delta(
                    reward, state, action, next_state, next_action, done
                )
                self.qfunction.update(state, action, delta)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards

    """ Calculate the delta for the update """
    def get_delta(self, reward, state, action, next_state, next_action, done):
        q_value = self.qfunction.get_q_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = (
            reward
            + (self.mdp.get_discount_factor() * next_state_value * (1 - done))
            - q_value
        )
        return delta

    """ Get the value of a state """
    def state_value(self, state, action):
        abstract
```
We inherit from this class to implement the Q-learning algorithm:

```python
from temporal_difference_learner import TemporalDifferenceLearner

class QLearning(TemporalDifferenceLearner):
    def state_value(self, state, action):
        max_q_value = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value
```
## SARSA: On-policy temporal difference learning
SARSA (State-action-reward-state-action) is an on-policy reinforcement learning algorithm. It is very similar to Q-learning, except that in its update rule, instead of estimate the future discount reward using $\max a\in A(s)Q(s',a)$, it actually selects the next action that it will execute, and updates using that instead. Taking this approach is known as **on-policy reinforcement learning**. Later in this section, we’ll discuss why this matters, but for now, let’s look at the SARSA algorithm and on-policy learning a bit more.
>[!info] Definition - On-policy and off-policy reinforcement learning
>Instead of estimating $Q(s',a')$ for the best estimated future state during update, **on-policy reinforcement learning** uses the actual next action to update.
>On-policy learning estimates $\mathcal{Q}^\pi(s,a)$ state action pairs, for the current behavior policy $\pi$, whereas **off-policy learning** estimates the policy independent of the current behavior.

The SARSA algorithm:
>[!todo] Algorithm 3 - SARSA
> **Input**: MDP $M=\left\langle S,s_0,A,P_a(s'a|s),r(s,a,s')\right\rangle$
> **Output**: Q-function $Q$
> Initialize $Q(s,a)\leftarrow0$ for all $s$ and $a$
> **repeat**
> $\qquad s\leftarrow$ the first state in episode $e$
>$\qquad$Select action $a$ to apply in $s$
> $\qquad$ **repeat** each step in episode $e$
> $\qquad\qquad$Execute action $a$ in state $s$
> $\qquad\qquad$Observe reward $r$ and new state $s'$
> $\qquad\qquad$Select action $a$ to apply in $s'$
> $\qquad\qquad \delta\leftarrow r+\gamma Q(s',a')-Q(s,a)$
> $\qquad\qquad Q(s,a)\leftarrow Q(s,a)+\alpha\delta$
> $\qquad\qquad s\leftarrow s'$
> $\qquad\qquad a\leftarrow a'$
> $\qquad s$  is the last state of episode $e$ (a terminal state)
> **until** $Q$ converges

The difference between the Q-learning and SARSA algorithms is in the update in the loop.
**Q-learning**:
- (1) selecting an action $a$
- (2) taking actions and observing the reward & next state $s'$
- (3) updating **optimistically** by assuming the future reward is $\max_{a'}Q(s',a')$
**SARSA**:
- (1) selecting action $a'$ for the *next* loop iteration
- (2) in the next iteration, taking action and observing the reward & next state $s'$
- (3) choosing $a'$ for the next iteration
- (4) updating using the estimation for the actual next chosen action.
There are two main differences:
- Q-learning will converge to the optimal policy irrelevant of the policy followed, because it is **off-policy**: it uses the greedy reward estimate in its update rather than following the policy such as $\epsilon$-greedy). Using a random policy, Q-learning will still converge to the optimal policy, but SARSA will not (necessarily).
- Q-learning learns an optimal policy, but this can be 'unsafe' or risky _during training_.
### Implementation
As with the Q-learning agent, we inherit from the `TemporalDifferenceLearner` class to implement SARSA. But the value of the next state V(s′) is calculated differently in the `SARSA` class:
```python
from temporal_difference_learner import TemporalDifferenceLearner

class SARSA(TemporalDifferenceLearner):
    def state_value(self, state, action):
        return self.qfunction.get_q_value(state, action)
```
## On-policy vs. off-policy
There are a few reasons why we have both on-policy and off-policy learning
### Learning from prior experience
The main advantage of **off-policy** approaches is that they can use samples from sources other than their own policy. Off-policy agents can be given a set of episodes of behavior from another agent, and can learn a policy by **demonstration**.
### Learning on the job
The main advantage of **on-policy** approaches is that they are useful for 'learning on the job', meaning that it is better for cases in which we want an agent to learn optimal behavior while operating in its environment.
### Combining off-policy and on-policy learning
### Evaluation and termination
With [[Value Iteration]], we terminate the algorithm once the improvement in value function reaches some threshold. However, in the *model-free environment*, we are not aiming to learn a complete policy - only enough to get us from the initial state to an absorbing state; or, in the case of an infinite MDP, to maximize rewards.
With model-free learning, we can instead evaluate the policy directly by **executing it and recording the reward we receive**. Then, we terminate when the policy has reached **convergence**. By convergence, we mean that the average cumulative reward of the policy is no longer increasing during learning.
## Limitations of Q-learning and SARSA
Two major limitations:
- Because we need to select the best action $a$ in Q-learning by iterating over all actions. This limits Q-learning to *discrete* action spaces.
- If we use a Q-table to represent Q-function, both state spaces and action spaces must be *discrete*, and further, they must be modest in size or the Q-table will become too large to fit into memory or at least so large that it will take many episodes to sample all state-action pairs.
## Takeaways
>[!success] Takeaways
>- **On-policy** reinforcement learning uses the action chosen by the policy for the update.
> - **Off-policy** reinforcement learning assumes that the next action chosen is the action that has the maximum Q-value, but this may not be the case because with some probability the algorithm will explore instead of exploit.
> - **Q-Learning** (off-policy) does it relative to the greedy policy.
> - **SARSA** (on-policy) learns action values relative to the policy it follows.
> - If we know the MDP, we can use model-based techniques:
> 	- **Offline**: Value Iteration
> 	- **Online**: [[Monte Carlo Tree Search]] and friends.
> - We can also use model-free techniques if we know the MDP model: we just sample transitions and observe rewards from the model.
> - If we do **not** know MDP, we need to use model-free techniques:
> 	- **Offline**: Q-learning, SARSA, and friends.

| Previous                |                              Next |
| :---------------------- | --------------------------------: |
| [[Multi-armed bandits]] | [[n-step reinforcement learning]] |
