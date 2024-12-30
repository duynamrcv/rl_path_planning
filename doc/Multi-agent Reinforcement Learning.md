>[!abstract] Learning outcomes
>- Define *"stochastic game"*
>- Explain the difference between single-agent reinforcement learning and multi-agent reinforcement learning

## Overview
Recall the idea of [[Monte-Carlo Tree Search#Foundation MDPs as ExpectiMax Trees|ExpectiMax trees]], which is the representation of MDPs. Recall that the white nodes are states and the black nodes are choices by the environment.
![[expectimax_tree.png]]
An extensive form game tree can be thought of as a slight modification to an ExpectiMax tree, except the choice at the black node is no longer left up to the ‘environment’, but is made by another agent instead.
![[extensive_form_game.png]]
An extensive form game can be solved with model-free reinforcement learning techniques and Monte-Carlo tree search techniques.
## Stochastic games
>[!info] Definition - Stochastic game
>A **Stochastic game** is a tuple $G=(S,s_0,A^1,...,A^n,r^1,...,r^n,P,\gamma)$
>- $S$ is a set of states
>- $s_0$ is the initial state
>- $A_j$ is the set of actions for agent $j$,
>- $P:S\times A_1\times...\times A_n\to\Omega(S)$ is a transition function that defines the probability of arriving in a state given a starting state and the actions chosen by all players
>- $r_j:S\times A_1\times...\times A_n\times S\to\mathbb{R}$ is the reward function for agent $j$
>- $\gamma$ is the discount factor

1. A stochastic game is a simply a version of an MDP in which the actions space and reward space are tuples: one action and one reward for each agent. It is, in fact, a generalization of MDPs because $n=1$ is a standard MDP.
2. A stochastic game is a generalization of an extensive form game. Express any extensive form game as a stochastic game; but not vice-versa because, for example, actions in extensive form games are deterministic and multiple agents cannot execute actions simultaneously. However, simulating an extensive form game by restricting all but one player to playing just a “null” action that has no effect.
## Solutions for stochastic games
In a stochastic game, the solution is a set of policies: one policy $\pi_j$ for each agent $j$. The **joint policy** of all agents is simply
$$\pi = [\pi^1, \ldots \pi^n]$$
The objective for an agent is to maximize its own expected discounted accumulated reward:
$$V^{\pi^j}(s) = E_{\pi^j}\left[\, \sum_{i} \gamma^i \, r^j(s_i, a, s_{i+1}) \ | \ s_0 = s, a = \pi(s_i)\right]$$
Note that $a=\pi(s_i)$ is the joint action of all agents. So, each agent’s objective is to maximize its own expected reward considering the possible actions of all other agents.
## Multi-agent Q-learning
Given the similarities between MDPs and stochastic games, the standard Q-learning algorithm can be considered as a multi-agent version.
>[!todo] Algorithm 1 - Multi-agent Q-learning
>**Input**: Stochastic game $G=(S,s_0,A^1,...,A^n,r^1,...,r^n,P,\gamma)$
>**Output**: Q-function $Q^j$ where $j$ is the agent
>Initialize $Q^j(s,a)=0$ for all state $s$ and joint actions $a$
>**repeat**
>$\qquad s\leftarrow$ the first state in episode $e$
>$\qquad$**for each** step in episode $e$
>$\qquad\qquad$Select action $a^j$ to apply in $s$ using $Q^j$ and a multi-armed bandit
>$\qquad\qquad$Execute action $a^j$ in state $s$
>$\qquad\qquad$Observer reward $r^j$ and new state $s'$
>$\qquad\qquad Q^j(s,a) \leftarrow Q^j(s,a) + \alpha \cdot [r^j + \gamma \cdot \max_{a'} Q^j(s',a') - Q^j(s,a)]$
>$\qquad\qquad s \leftarrow s'$
>$\qquad$**until** the end of episode $e$ (a terminal state)
>**until** $Q$ converges

### Opponent moves
How should choosing actions for opponents' moves? There are a few ways:
1. **Random selection**: Select a random action. This is easy, but it means that we may end up exploring a lot of actions that will never be taken by a good opponent and that lead to poor Q-values for actions.
2. **Using a fixed policy**: Using an existing **stochastic** policy that gives reasonable behavior of the opponent. This could be hand-coded or learnt from a similar game.
3. **Self play**: Simultaneously learning a policy for both ourselves and our opponents, and choosing actions for opponents based on the learnt policies. If our actions spaces are the same, a single policy can be learnt for both ourselves and our opponents.
## Multi-agent Monte-Carlo tree search
Similar to Q-learning, [[Monte-Carlo tree search]] to the multi-agent case. Multi-agent MCTS is similar to single-agent MCTS, with the simple modification:
1. **Selection**: For ‘our’ moves, we run selection as before, however, we also need to select models for our opponents. In multi-agent MCTS, an easy way to do this is via *self-play*. Each node has a player whose turn it is, and we use a multi-armed bandit algorithm to choose an action for that player.
2. **Expansion**: Instead of expanding a child node based on $P_a(s'|s)$, we expand one of our actions, which leads to our opponents’ node. In the next iteration, that node becomes a node of expansion.
3. **Simulate**: We then simulate as before, and we learn the rewards when we receive them. Recall that the rewards are a vector of rewards: one for each player.
4. **Backpropagate**: The backpropagation step is the same as before, except that we need to keep the value of the node for every player.
## Takeaways
>[!success] Takeaways
>- For solving extensive form games in a model-free or simulated environment, we can extend techniques like Q-learning, SARSA, and MCTS from single-agent to multi-agent environments in a straightforward way.
>- Other more sophisticated techniques exist for true stochastic games (with simultaneous moves), such as **mean field Q-learning**, which reduces the number of possible interactions between actions by approximating joint actions between all pairs of actions, rather than all global interactions among the agents.

|               Previous |
| ---------------------: |
| [[Backward induction]] |
