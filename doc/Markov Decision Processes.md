> [!abstract] Learning outcomes
> - Define **Markov Decision Process**
> - Identify situation in which MDPs are a suitable model of a problem
> - Compare MDPs to models of search, such as classical planning and heuristic search
> - Explain how Bellman equations are solutions to MDP problems
## Overview
A **Markov Decision Process** (MDPs) is a framework for describing sequential decision making problems. In reinforcement learning, decision maker is **agent**, and decisions are **actions** that they execute in an **environment** or **state**.
MDPs consider **stochastic** non-determinism, where there is a probability distribution over outcomes, differ from heuristic search and classical planning algorithms assume action are **deterministic**.

## Markov Decision Processes
> [!info] Definition - Markov Decision Process
> A **Markov Decision Process** (MDP) is a **fully observable**, **probabilistic** state model. The common formulation of MDPs is a **Discounted-Reward Markov Decision Process**. A discount-reward MDP is a tupe $(S,s_0,A,P,r,\gamma)$ containing:
> - a state space $S$;
> - initial state $s_0\in S$;
> - actions $A(s)\subseteq A$ applicable in each state $s\in S$ that agent can execute;
> - **transition probabilities** $P_a(s'|s)$ for $s\in S$ and $a\in A(s)$;
> - **rewards** $r(s,a,s')$ positive or negative of transitioning from state $s$ to state $s'$ using action $a$;
> - a **discount factor** $0\leq\gamma<1$.

**States** are the possible situations in which the agent can be in. Each state captures the information required to make a decision.
**State space** is simple the set of all possible states.
**Actions** allow agents to affect the environments/state, transition from one state to another.
**Transition probabilities** is effect of each action, including the probabilities of each outcome.
**Reward** specifies the benefit or cost of executing an action in a state.
**Discount factor** $\gamma$ determines how much a future reward should be discounted compared to a current reward.
The abstract of an MDP:
```python
class MDP:
    """ Return all states of this MDP """
    def get_states(self):
        abstract

    """ Return all actions with non-zero probability from this state """
    def get_actions(self, state):
        abstract

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """
    def get_transitions(self, state, action):
        abstract

    """ Return the reward for transitioning from state to
        nextState via action
    """
    def get_reward(self, state, action, next_state):
        abstract

    """ Return true if and only if state is a terminal state of this MDP """
    def is_terminal(self, state):
        abstract

    """ Return the discount factor for this MDP """
    def get_discount_factor(self):
        abstract

    """ Return the initial state of this MDP """
    def get_initial_state(self):
        abstract

    """ Return all goal states of this MDP """
    def get_goal_states(self):
        abstract
```
### MDPs vs deterministic search
There are four main differences:
- The transition function is not deterministic. Each action has a probability of $P_a(s'|s)$ of ending  in state $s'$ if $a$ is executed in the state $s$, whereas in classical planning, the outcome of each action is known.
- There are no goal states. Each action receives a reward when applied. The value of reward is dependent on the sate in which it is applied.
- There are no action costs. Action costs are modeled as negative rewards.
- It have a discount factor. In classical planning, executing an action typically has a cost. A discount factor of less than 1 implicitly rewards shorter plans.
## Policies
Instead of a sequence of actions, an MDP produces a *policy*.
> [!info] Definition - Policy
> A **policy** $\pi$ is a function that presents the best action to choose in each state. A polity can be **deterministic** or **stochastic**.

### Deterministic vs. stochastic policies
A **deterministic policy** $\pi:S\to A$ is a function that maps stats to actions. It specifies which action $\pi(s)$ to choose in ever possible state $s$. The output from a planning algorithm would be a dictionary-like object or a function that takes a state and return an action.
A **stochastic policy** $\pi:S\times A\to\mathbb{R}$ specifies the **probability distribution** from which an agent should select an action. $\pi(s,a)$ specifies the probability that action $a$ should be executed in state $s$. The action with the maximum $\pi(s,a)$ will be taken.
![[policy.png]]
The figure show the different.
- The output of a deterministic policy is an action. It always return the same action in the same state
- The output of a stochastic policy is a probability distribution over the set of possible actions. E.g. action $b$ would be the most likely to be chosen.
### Representing policies
Policies can be represented in several ways, but have the same basic interface.
- *Deterministic policy*: the ability to update the policy and the ability to get an action for a state.
- *Stochastic policy*: get the value or probability of playing an action.
```python
class Policy:
    def select_action(self, state, action):
        abstract

class DeterministicPolicy(Policy):
    def update(self, state, action):
        abstract

class StochasticPolicy(Policy):
    def update(self, states, actions, rewards):
        abstract

    def get_probability(self, state, action):
        abstract
```
The simplest way to represent a policy is a tabular policy, which keeps a table that maps from each state to the action for that state.
## Optimal Solutions for MDPs
For discounted-reward MDPs, optimal solutions maximize the **expected discounted accumulated reward** from the initial state $s_0$.
> [!info] Definition - Expected discount reward
> The **expected discounted reward** from $s$ for a policy $\pi$ is:
> $$V^\pi(s)=E_\pi\left[\sum_i{\gamma^ir(s_i,a_i,s_{i+1})}|s_0=s,a_i=\pi(s_i)\right]$$
> So, $V^\pi(s)$ defines the expected value of following policy $\pi$ from state $s$.

> [!info] Definition - Bellman equation
> The **Bellman equation** describes the condition that must hold for a policy to be optimal, and is defined recursively as:
> $$V(s)=\max_{a\in A(S)}\sum_{s'\in S}{P_s(s'|s)\left[r(s,a,s')+\gamma V(s')\right]}$$

Therefore, $V$ is optimal if for all states $s$, $V(s)$ describes the total discounted reward for taking the action with the highest reward over an indefinite/infinite horizon.
$$V(s) = \overbrace{\max_{a \in A(s)}}^{\text{best action from $s$}} \overbrace{\underbrace{\sum_{s' \in S}}_{\text{for every state}} P_a(s' \mid s) [\underbrace{r(s,a,s')}_{\text{immediate reward}} + \underbrace{\gamma}_{\text{discount factor}} \cdot  \underbrace{V(s')}_{\text{value of } s'}]}^{\text{expected reward of executing action $a$ in state $s$}}$$
- The *reward* of an action is: the sum of *immediate reward* for all states possibly plus the *discounted future reward* of those states.
- The *discounted future reward* is *discount factor* $\gamma$ times the value of $s'$
- Due to multiple states, the reward multiple by the *probability* of it happening $P_a(s'|s)$.
Bellman equations can be described slightly differently, named _Q-values_.
If $V(s)$ is the expected value of being in state $s$ and acting optimally according to policy, the *Q-value* can be used in a state $s$ choosing action $a$.
> [!info] Definition - Q-value
> The **Q-value** for action $a$ in state $s$ is defined as:
> $$Q(s,a)=\sum_{s'\in S}{P_a(s'|s)\left[r(s,a,s')+\gamma V(s')\right]}$$
> This represents the value of choosing action $a$ in state $s$ and then following this same policy until termination.

Using this, the Bellman equation is then defined as:
$$V(s)=\max_{a\in A(s)}Q(s,a)$$
## Policy extraction
Given a value function $V$, select the action that maximizes expected utility. If value function $V$ is optimal, the action with the highest expected reward can be selected:
$$\pi(s)=\arg\max_{a\in A(s)}\sum_{s'\in S}P_a(s'|s)\left[r(s,a,s')+\gamma V(s')\right]$$
This can be called **policy extraction**. Alternatively, given a Q-function, we can use:
$$\pi(s)=\arg\max_{a\in A(s)}Q(s,a)$$
This is simpler than using the value functions, because not need to sum over the set of possible output states, but need to store $|A|\times|S|$ value in a Q-function, but just $|S|$ values in a value function.
## Partially observable MDPs
MDPs assume that agent always knows exactly what state it is in (fully observable). However, this is not valid for many tasks.
> [!info] Definition - Partially-observable MDP
> **Partially-observable MDPs** (POMDPs) relax the assumption of full-observability. A POMDP is defined as:
> - state $s\in S$
> - actions $A(s)\subseteq A$
> - transition probabilities $P_a(s'|s)$ for $s\in S$ and $a\in A(s)$
> - initial **belief state** $b_0$
> - reward function $r(s,a,s')$
> - a set of possible observations $Obs$
> - a **sensor model** given by probabilities $O_a(o|s),o\in Obs$

The sensor model allows the agent to observe the environment. If an agent executes an action $a$, it has probability $O_a(a|s')$ of observing state $s'$.
Solving POMDPs is similar to solving MDPs. The only difference is that POMDP is a standard MDP with a new state space: each state is a **probability distribution** over the set $S$, a.k.a a **belief state**, which defined the probability of being in each state $S$. Lead to harder to solve.
Like MDPs, solutions are policies that map belief state to actions. Optimal policies maximize the expected reward.

---
> [!success] Takeaways
> - **Markov Decision Processes** (MDPs) model sequential decision-making problems in which the outcome of an action is stochastic; although the agent can observe the state once the action is executed.
> - The solution to an MDP is a **policy**.
> - A **deterministic policy** is a mapping from states to actions. For policy $π$, the term $π(s)$ returns the action to execute in state $s$.
> \- A **stochastic policy** is a mapping from state actions pairs to probabilities. For policy $π$, the term $π(s,a)$ returns the probability of selecting action $a$ in state $s$.
> - An **optimal policy** maximizes the expected discounted reward.
> - The **Bellman equation** describes the condition that must hold for a policy to be optimal.

| Previous                                   |                    Next |
| ------------------------------------------ | ----------------------: |
| [[Introduction to Reinforcement Learning]] | [[Value-based methods]] |
