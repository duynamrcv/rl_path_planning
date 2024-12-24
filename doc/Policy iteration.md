>[!abstract] Learning outcomes
>- Apply policy iteration to solve small-scale MDP problems manually and program iteration algorithms to solve medium-scale MDP problems automatically.
>- Discuss the strengths and weaknesses of policy iteration
>- Compare and contrast policy iteration to value iteration

## Overview
The other common way that MDPs are solved is using **policy iteration** – an approach that is similar to value iteration. While value iteration iterates over value functions, policy iteration iterates over policies themselves, creating a strictly improved policy in each iteration (except if the iterated policy is already optimal).
Policy iteration first starts with some (non-optimal) policy, such as a random policy, and then calculates the value of each state of the MDP given that policy — this step is called **policy evaluation**. It then updates the policy itself for every state by calculating the expected reward of each action applicable from that state.
## Policy evaluation
The concept in policy iteration is **policy evaluation**, which is an evaluation of the expected reward of a policy. The **expected reward** of policy $\pi$ from $s$, $V^\pi(s)$, is the weighted average of reward of the possible state sequence defined by policy times their probability $\pi$.
>[!info] Definition - Policy evaluation
> _Policy evaluation_ can be characterized as $V^\pi(s)$ as defined by the following equation:
> $$V^\pi(s) =  \sum_{s' \in S} P_{\pi(s)} (s' \mid s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]$$
> where $V^\pi(s)=0$ for terminal states.

Note that this is very similar to the [[Markov Decision Processes#Optimal Solutions for MDPs|bellman equation]], except $V^\pi(s)$ is not the value of the best action, but instead just as the value for $\pi(s)$, the action that would be chosen in s by the policy $\pi$. Note the expression $P_{\pi(s)}(s'|s)$ instead of $P_a(s'|s)$, which means we only evaluate the action that the policy defines.
>[!todo] Algorithm 1 - Policy evaluation
>**Input:** The policy $\pi$ for evaluation, value function $V^pi$, and MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
>**Output:** Value function $V^\pi$
>**repeat**
>$\qquad\delta\leftarrow0$
>$\qquad$**for each** $s\in S$
>$\qquad\qquad V^{'\pi}(s)\leftarrow\sum_{s'\in S}P_{\pi(s)}(s'|s)\left[r(s,a,s')+\gamma V^\pi(s')\right]$
>$\qquad\qquad\Delta\leftarrow\max(\Delta,|V^{'\pi}(s)-V^\pi(s)|)$
>$\qquad V^\pi\leftarrow V'^{\pi}$
>**until** $\Delta\leq\theta$

The **optimal expected reward** $V^*(s)$ is $\max_\pi V^\pi(s)$ and the **optimal policy** is $\arg\max_\pi V^\pi(s)$.
## Policy improvement
**Policy improvement** is used to change the policy by updating the actions based on $V(s)$ received from the policy evaluation. Let $Q^\pi(s,a)$ be the expected reward from $s$ when doing a first and then following the policy $\pi$.
$$Q^{\pi}(s,a)  =  \sum_{s' \in S} P_a(s' \mid s)\ [r(s,a,s') \, + \,  \gamma\ V^{\pi}(s')]$$
In this case, $V^\pi(s')$ is the value function from the policy evaluation. If there is an action a such that $Q^\pi(s,a)>Q^\pi(s,\pi(s))$, then the policy $\pi$ can be **strictly improved** by setting $\pi(s)\leftarrow a$. This will improve the overall policy.
## Policy iteration
Define the **policy iteration**, which computes an optimal $\pi$ by performing a sequence of interleaved policy evaluations and improvements.
>[!todo] Algorithm 2 - Policy iteration
>**Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
>**Output**: Policy $\pi$
>Set $V^\pi(s)=0$ for all $s$ to value function
>Set $\pi(s)=0$ for all $s$ to policy, where $a\in A$ is an action
>**repeat**
>$\qquad$Compute $V^\pi(s)$ fro all $s$ using policy evaluation
>$\qquad$**for each** $s\in S$
>$\qquad\qquad\pi(s)\leftarrow\arg\max_{a\in A(s)}Q^\pi(s,a)$
>**until** $\pi$ does not change

The policy iteration algorithm finishes with an optimal $π$ after a finite number of iterations, because the number of policies is finite, bounded by $O(|A|^{|S|})$, unlike value iteration, which can theoretically require infinite iterations. However, each iteration costs $O(|S|^2|A|+|S|^3)$. Empirical evidence suggests that the most efficient is dependent on the particular MDP model being solved, but that surprisingly few iterations are often required for policy iteration.
## Implementation
```python
from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from qtable import QTable


class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):
        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                new_value = values.get_q_value(
                    self.mdp, state, policy.select_action(state, actions)
                )
                values.add(state, new_value)
                delta = max(delta, abs(old_value - new_value))

            # terminate if the value function has converged
            if delta < theta:
                break
        return values

    """ Implmentation of policy iteration iteration. Returns the number of iterations executed """

    def policy_iteration(self, max_iterations=100, theta=0.001):
        # create a value function to hold details
        values = TabularValueFunction()
        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():

                actions = self.mdp.get_actions(state)
                old_action = self.policy.select_action(state, actions)

                q_values = QTable(alpha=1.0)
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)
                # V(s) = argmax_a Q(s,a)
                new_action = q_values.get_argmax_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)
                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            if not policy_changed:
                return i
        return max_iterations
```
## Takeaways
>[!success] Takeaways
>- **Policy iteration** is a dynamic programming technique for calculating a policy directly, rather than calculating an optimal $V(s)$ and extracting a policy; but one that uses the concept of values.
>- It produces an optimal policy in a finite number of steps.
>- Similar to value iteration, for medium-scale problems, it works well, but as the state-space grows, it does not scale well.

| Previous                 |                 Next |
| :----------------------- | -------------------: |
| [[Policy-based methods]] | [[Policy gradients]] |
