>[!abstract] Learning outcomes
>- Apply actor-critic methods to solve small-scale MDP problems manually and program actor critic algorithms to solve medium-scale MDP problems automatically.
>- Compare and contrast actor-critic methods with policy gradient methods like REINFORCE and value-based reinforcement learning.

The sample efficiency problem in REINFORCE leads to issues with policy convergence. As with Monte-Carlo simulation, the high variance in the cumulative rewards $G$ over episodes leads to instability. **Actor critic** methods aim to mitigate this problem. The idea is that instead of learning a value function or a policy, we learn both. The policy is called the **actor** and the value function is called the **critic**. The primary idea is that the actor produces actions, and as in [[Temporal difference reinforcement learning#Temporal difference (TD) reinforcement learning|temporal difference learning]], the value function (the critic) provides feedback or "criticism" about these actions as a way of bootstrapping.
![[actor-critic_framework.png]]
The figure gives an abstract overview of actor-critic frameworks - in this case, Q actor-critic. As with REINFORCE, actions are samples from the stochastic policy $\pi_\theta$. Given the next action, we update the actor (the policy) and then the critic (the value function or Q function). The selected action is executed in the environment, and the agent receives the reward and next state observation.
## Q Actor-Critic
The **Q Actor-Critic** algorithm uses a Q-function as the critic:
>[!todo] Algorithm 1 - Q Actor-Critic
>**Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$, a differentiable actor policy $\pi_\theta(s,a)$, a differentiable critic Q-function $Q_w(s,a)$
>**Output**: Policy $\pi_\theta(s,a)$
>Initialize actor $\pi$, parameters $\theta$ and critic parameters $w$
>**repeat** for each episode $e$
>$\qquad s\leftarrow$ the first state in episode $e$
>$\qquad$Select action $a\sim\pi_\theta(s,a)$
>$\qquad$**repeat** for each step in episode $e$
>$\qquad\qquad$Execute action $a$ in state $s$
>$\qquad\qquad$Observed reward $r$ and new state $s'$
>$\qquad\qquad$Select action $a'\sim\pi_\theta(s',a')$
>$\qquad\qquad\delta\leftarrow\gamma\cdot Q_w(s',a')-Q_w(s,a)$
>$\qquad\qquad w\leftarrow w+\alpha_w\cdot\delta\cdot\nabla Q_w(s,a)$
>$\qquad\qquad\theta\leftarrow\theta+\alpha_\theta\cdot\delta\cdot\nabla\ln\pi_\theta(s,a)$
>$\qquad\qquad s\leftarrow s';a\leftarrow a'$
>$\qquad$**until** $s$ is the last state of episode $e$ (a terminal state)
>**until** $\pi_\theta$ converges

Note that we have two different learning rates $\alpha_w$ and $\alpha_\theta$ for the Q-function and policy respectively. This algorithm simultaneously learns the policy (actor) $\pi_\theta$ and a critic (Q-function) $Q_w$, but the critic is learnt only to provide the temporal difference update, not to extract the policy.
The reason the actor critic methods still work like this is because the actor policy $\pi_\theta$ selects actions, while the critic $Q_w(s,a)$ is only ever used to calculate the temporal difference estimate for an already selected action. We do not have to iterate over the critic Q-function to select actions, so we do not have to iterate over the set of actions – we just use the policy. As such, this will still extend to continuous and large state spaces and be more efficient for large action space.
## Implementation
The `ActorCritic` class is an abstract class that looks similar to that of `QLearning`, except that we update both the actor and the critic:
```python
import statistics
from itertools import count
from model_free_learner import ModelFreeLearner

class ActorCritic(ModelFreeLearner):
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100, max_episode_length=float("inf")):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            deltas = []

            state = self.mdp.get_initial_state()
            action = self.actor.select_action(state, self.mdp.get_actions(state))
            episode_reward = 0.0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                next_action = self.actor.select_action(
                    next_state, self.mdp.get_actions(next_state)
                )

                delta = self.get_delta(
                    reward, state, action, next_state, next_action, done
                )

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                deltas.append(delta)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length:
                    break

            self.update_critic(states, actions, deltas)
            self.update_actor(states, actions, deltas)

            episode_rewards.append(episode_reward)

        return episode_rewards

    def get_delta(self, reward, state, action, next_state, next_action, done):
        q_value = self.state_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = (
            reward
            + (self.mdp.get_discount_factor() * next_state_value * (1 - done))
            - q_value
        )
        return delta

    def update_actor(self, states, actions, deltas):
        abstract

    def update_critic(self, states, actions, deltas):
        abstract
```
Next, we have to instantiate the `ActorCritic` class as a `QActorCritic` class to implement the `update_actor` and `update_critic` classes:
```python
from actor_critic import ActorCritic


class QActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, states, actions, deltas):
        self.actor.update(states, actions, deltas)

    def update_critic(self, states, actions, deltas):
        self.critic.batch_update(states, actions, deltas)

    def state_value(self, state, action):
        return self.critic.get_q_value(state, action)
```
## Takeaways
>[!success] Takeaways
>- Like REINFORCE, **actor-critic** methods are policy-gradient based, so directly learn a policy instead of first learning a value function or Q-function.
>- Actor-critic methods also learn a **value function or Q-function** to reduce the variance in the cumulative rewards.

| Previous             |                                   Next |
| :------------------- | -------------------------------------: |
| [[Policy gradients]] | [[Modelling and abstraction for MDPs]] |
