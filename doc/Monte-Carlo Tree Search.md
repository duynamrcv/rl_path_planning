>[abstract] Learning outcomes
>- Explain the difference between offline and online learning for MDPs
>- Apply MCTS solve small-scale MDP problems and program MCTS algorithsm to solve medium-scale MDP problem
>- Construct a policy from Q-functions resulting from MTCS algorithms
>- Integrate multi-armed bandit algorithms to MCTS algorithms
>- Compare and contrast MCTS to value iteration
>- Discuss the strengths and weaknesses of the MCTS algorithms

## Offline planning & Online planning for MDPs
Value iteration method is an **offline** planning method because it solves problems offline for all possible states, and then use the solution online to act.
The state space $S$ is usually *far* to big to determine $V(s)$ or $\pi$ exactly.
In **online** planning, planning is undertaken immediately before executing an action. Once an action is executed, planning again from the new state. As such, planning and execution are interleaved:
- For each state $s$ visited, the set of all available action $A(s)$  partially evaluated
- The quality of each action $a$ is approximated by averaging the expected reward of trajectories over $S$ obtained by repeated simulations, giving as an approximation for $Q(s,a)$.
- The chosen action is $\arg\max_{a'}Q(s,a)$.
In online planning, need a **simulator** that approximates the transitions function $P_a(s'|s)$ and reward function $r$ of MDP. The simulator allows to run repeated simulations of possible futures to gain an idea of what moves are the good moves compared to others.
## Overview
Monte Carlo Tree Search (MTCS) is a name for a _set_ of algorithms all based around the same idea. Here, focusing on using an algorithm for solving single-agent MDPs in a model-based manner. 
### Foundation: MDPs as ExpectiMax Trees
To get the idea of MCTS, we note that MDPs can be represented as trees (or graphs), called **ExpectiMax** trees:![[expectimax_tree.png]]
The letter $a-e$ represent actions, and letters $s-x$ represent states. White nodes are state nodes, and the small black nodes represent the probabilistic uncertainty: the 'environment' choosing which outcome from an action happens, based on the transition function.
### Monte Carlo Tree Search - Overview
The algorithm is online, which means the action selection is interleaved with action execution. Fundamental features:
- The Q-value $Q(s,a)$ for each is approximated using **random simulation**.
- For a single-agent problem, an ExpectiMax **search tree** is built incrementally.
- The search terminates when some pre-defined computational budget is used up, such as a time limit or a number of expanded nodes. Therefore, it is an **any time** algorithm, as it can be terminated at any time and still give an answer.
- The best performing action is returned
	- This is complete if there are no dead-ends
	- This is optimal if an entire search can be performed (unusual)
## The framework: Monte Carlo Tree Search (MCTS)
The basic framework is to build up a tree using simulation. The states that have been evaluated are stored in a search tree. The set of evaluated state is incrementally built be iterating over the following four steps:
- **Selection**: Select a single node in the tree that is **not fully expanded**. At least one of its children is not yet explored.
- **Expand**: Expand this node by applying one available action fro the node.
- **Simulate**: From one of the outcomes of the expanded, perform a complete random simulation of the MDP to a terminating state. Assume that the simulation is finite.
- **Backpropagate**: Finally, the value of the node is backpropagated to the root node, updating the value of each ancestor node on the way using expected value.
### Selection
Start at the root node, and successively **select a child node** until we reach a node that is not fully expanded.
![[MCTS_selection.png]]
### Expansion
Unless the node ending up at is a terminating state,  **expand the children** of the selected node by choosing an action and creating new nodes using the action outcomes.
![[MCTS_expansion.png]]
### Simulation
Choose one of the new nodes and perform a **random simulation** of the MDP to the terminating state.
![[MCTS_simulation.png]]
### Backpropagation
Given the reward $r$ at the terminating state, **backpropagate** the reward to calculate the value $V(s)$ at each state along the path.
## Algorithm
Basically, MCTS algorithm is incrementally built of the search tree. Each node in the tree stores:
- a set of children nodes;
- pointers to its parent node and parent action; and
- the number of times it has been visited.
This tree is used to explore different Monte-Carlo simulations to learn a Q-function $Q$.
>[!todo] Algorithm 1 - Monte-Carlo Tree Search
>**Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$, Q-function $Q$, time limit $T$
>**Output**: updated Q-function $Q$
>**while** current_time $< T$ **do**
>$\qquad$ selected_node $\leftarrow$ Select($s_0$)
>$\qquad$ child $\leftarrow$ Expand(selected_node) -- expand and choose a child node to simulate
>$\qquad G\leftarrow$ Simulate(child) -- simulate from child
>$\qquad$ Backpropagate(selected_node, child, $Q$, $G$)
>**return** $Q$

There are four main parts to the algorithm:
- **Selection**:The first loop selects a branch in the tree using a multi-armed bandit algorithm using $Q(s,a)$. The outcome occurs from an action is chosen according to $P(s'|s)$ defined in the MDP
> [!todo] Algorithm 2 - Function - Select($s:S$)
> **Input**: state $s$
> **Output**: unexpanded state $s$
> **while** $s$ is fully expanded **do**
> $\qquad$Select action $a$ to apply in $s$ using a multi-armed bandit algorithm 
> $\qquad$Choose one outcomes $s'$ according to $P_a(s'|s)$
> $\qquad s\leftarrow s'$
> **return** $s$

- **Expansion**: Select an action $a$ to apply in state $s$, either randomly or using an heuristic. Get an outcome state $s′$ from applying action $a$ in state $s$ according to the probability distribution $P(s'∣s)$. Expand a new environment node and a new state node for that outcome.
>[!todo] Algorithm 3 - Function - Expand($s:S$)
>**Input**: state $s$
>**Output**: expanded state $s'$
>**if** $s$ is fully expanded **then**
>$\qquad$Randomly select action $a$ to apply in $s$
>$\qquad$Expand one outcome $s'$ according to $P_a(s'|s)$ and observe reward $r$
>**return** $s'$

- **Simulation**: Perform a randomized simulation of the MDP until we reach a terminating state. That is, at each choice point, randomly select an possible action from the MDP, and use transition probabilities $P_a(s'∣s)$ to choose an outcome for each action. Heuristics can be used to improve the random simulation by guiding it towards more promising states. $G$ is the cumulative discounted reward received from the simulation starting at $s'$ until the simulation terminates. To avoid memory explosion, discard all nodes generated from the simulation.
- **Backpropagation**: The reward from the simulation is backpropagated from the selected node to its ancestors recursively. Must not forget the discount factor! For each state $s$ and action $a$ selected in the Select step, update the cumulative reward of that state.
>[!todo] Algorithm 4 - Function - Backpropagation($s:S;a:A;Q:S\times A\to\mathbb{R};G:\mathbb{R}$)
>**Input**: state-action pair $(s,a)$, Q-function $Q$, reward $G$
>**Output**: none
>**do**
>$\qquad N(s,a)\leftarrow N(s,a)+1$
>$\qquad G\leftarrow r+\gamma G$
>$\qquad Q(s,a)\leftarrow Q(s,a)+\dfrac{1}{N(s,a)}\left[G-Q(s,a)\right]$
>$\qquad s\leftarrow$ parent of $s$
>$\qquad a\leftarrow$ parent action of $s$
>**while** $s\neq s_0$

Because action outcomes are selected according to $P_a(s'|s)$, this will converge to the average expected reward. This is why the tree is called an **ExpectiMax** tree: maximize the expected return.
**But:** what if we do not know $Pa(s'|s)$?
Provided that we can **simulate** the outcomes; e.g. using a code-based simulator, then this does not matter. Over many simulations, the Select (and Expand/Execute steps) will sample $Pa(s'|s)$ sufficiently close that $Q(s,a)$ will converge to the average expected reward. Note that this is **not a model-free** approach: we still need a model in the form of a simulator, but we do not need to have explicit transition and reward functions.
## Action selection
Once running out of computational time, the action is selected that maximizes the expected return, which is simply the one with the highest Q-value from our simulations:
$$\arg\max_a\in A(s)Q(s_0,a)$$
Executing the action and waiting to see which outcome occurs for the action. Once the outcome state $s'$ is returned, the process is all over again, except with $s_0\leftarrow s'$.
However, importantly, we can _keep_ the sub-tree from state $s'$, as we already have done simulations from that state. We discard the rest of the tree and incrementally build from $s'$.
## Upper Confidence Trees (UCT)
When selecting nodes using [[Multi-armed bandits]], the UCB1 is proven to be successful in MCTS. The **Upper Confidence Trees** (UTC) algorithm is the combination of MCTS with UCB1 for selecting the next node:
$$UCT = MCTS+UCB1$$
The UCT selection strategy is similar to the UCB1:
$$\text{argmax}_{a \in A(s)} Q(s,a) + 2 C_p \sqrt{\frac{2 \ln N(s)}{N(s,a)}}$$
$N(s)$ is the number of times a state node has been visited, and $N(s,a)$ is the number of times a has been selected from this node. $C_p>0$ is the exploration constant, which determines can be increased to encourage more exploration, and decreased to encourage less exploration. Ties are broken randomly.
>[!note]
>If $Q(s,a)\in[0,1]$ and $C_p=\dfrac{1}{\sqrt{2}}$ then in two-player zero-sum, UTC converges to the well-known Mimimax algorithm.

## Implementation
First, we create a class `Node`, which forms the basis for the tree:
```python
import math
import time
import random
from collections import defaultdict


class Node:
    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """
    def select(self): abstract
    
    """ Expand a node if it is not a terminal node """
    def expand(self): abstract

    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child): abstract

    """ Return the value of this node """
    def get_value(self):
        max_q_value = self.qfunction.get_max_q(
            self.state, self.mdp.get_actions(self.state)
        )
        return max_q_value

    """ Get the number of visits to this state """
    def get_visits(self):
        return Node.visits[self.state]

class MCTS:
    def __init__(self, mdp, qfunction, bandit):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """
    def mcts(self, timeout=1, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()

        start_time = time.time()
        current_time = time.time()
        while current_time < start_time + timeout:
            # Find a state node to expand
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node):
                child = selected_node.expand()
                reward = self.simulate(child)
                selected_node.back_propagate(reward, child)
            current_time = time.time()
        return root_node

    """ Create a root node representing an initial state """
    def create_root_node(self): abstract

    """ Choose a random action. Heustics can be used here to improve simulations. """
    def choose(self, state):
        return random.choice(self.mdp.get_actions(state))

    """ Simulate until a terminal state """
    def simulate(self, node):
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        while not self.mdp.is_terminal(state):
            # Choose an action to execute
            action = self.choose(state)

            # Execute the action
            (next_state, reward, done) = self.mdp.execute(state, action)

            # Discount the reward
            cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward
            depth += 1
            state = next_state
        return cumulative_reward
```
The select, expand, and backpropagate methods in these two node classes:
```python
import random
from mcts import Node
from mcts import MCTS

class SingleAgentNode(Node):
    def __init__(
        self,
        mdp,
        parent,
        state,
        qfunction,
        bandit,
        reward=0.0,
        action=None,
    ):
        super().__init__(mdp, parent, state, qfunction, bandit, reward, action)

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return true if and only if all child actions have been expanded """
    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """
    def select(self):
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction)
            return self.get_outcome_child(action).select()

    """ Expand a node if it is not a terminal node """
    def expand(self):
        if not self.mdp.is_terminal(self.state):
            # Randomly select an unexpanded action to expand
            actions = self.mdp.get_actions(self.state) - self.children.keys()
            action = random.choice(list(actions))

            self.children[action] = []
            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[self.state] = Node.visits[self.state] + 1
        Node.visits[(self.state, action)] = Node.visits[(self.state, action)] + 1

        q_value = self.qfunction.get_q_value(self.state, action)
        delta = (1 / (Node.visits[(self.state, action)])) * (
            reward - self.qfunction.get_q_value(self.state, action)
        )
        self.qfunction.update(self.state, action, delta)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node """
    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        (next_state, reward, done) = self.mdp.execute(self.state, action)

        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            if next_state == child.state:
                return child

        # This outcome has not occured from this state-action pair previously
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
        )

        # Find the probability of this outcome (only possible for model-based) for visualising tree
        probability = 0.0
        for (outcome, probability) in self.mdp.get_transitions(self.state, action):
            if outcome == next_state:
                self.children[action] += [(new_child, probability)]
                return new_child

class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return SingleAgentNode(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
        )
```
## Function approximation
As with standard Q-learning, [[Q-function approximation]] can be used to generalize learning in MCTS. In particular, an off-line method such as Q-learning or SARSA with Q-function approximation can be used to learn a general Q-function. Often this will work quite well, however, the issue with Q-function approximation is sometimes the approximation does not work well for certain states during execution.
To mitigate this, we can then use MCTS (online planning) to search from the actual state, but starting with the pre-trained Q-function. This has two benefits:
- The MCTS supplements the pre-trained Q-function by running simulations from the actual initial state $s_0$, which may reflect the real rewards more accurately than the pre-trained Q-function given that it is an approximation.
- The pre-trained Q-function improves the MCTS search by guiding the selection step. In effect, the early simulations are not as random because there is some signal to use. This helps to mitigate the **cold start** problem, which is when we have no information to exploit at the start of learning.
## Why does it work so well (sometimes)?
It addresses exploitation vs. exploration comprehensively.
- UCT is **systematic**:
    - Policy evaluation is **exhaustive** up to a certain depth.
    - Exploration aims to **minimize regret**.
### Value iteration vs. MCTS
Often the set of states reachable from the initial state $s_0$ using an optimal policy is much smaller than the set of total states. In this regards, value iteration is exhaustive: it calculates behavior from states that will never be encountered if we know the initial state of the problem.
MCTS (and other search methods) methods thus can be used by just taking samples starting at $s_0$. However, the result is not as general as using value/policy iteration: the resulting solution will work only from the known initial state $s_0$ or any state reachable from $s_0$ using actions defined in the model. Whereas value iteration works from any state.

|                      |        Value iteration        |                                    MCTS                                     |
| -------------------- | :---------------------------: | :-------------------------------------------------------------------------: |
| Cost                 |   Higher cost (exhaustive)    |             Lower cost (does not solve for entire state space)              |
| Coverage/ Robustness | Higher (works from any state) | Lower (works only from initial state or state reachable from initial state) |
This is important: *value iteration* is then more expensive for many problems, however, for an agent operating in its environment, we only solve exhaustively once, and we can use the resulting policy many times no matter state we are in latter. For *MCTS*, we need to solve **online** each time we encounter a state we have not considered before. 
## Takeaways
>[!success] Takeways
>- **Monte Carlo Tree Search** (MCTS) is an anytime search algorithm, especially good for stochastic domains, such as MDPs.
>	- It can be used for model-based or simulation-based problems.
>	- Smart selection strategies are _crucial_ for good performance.
>- **UCT** is the combination of MCTS and UCB1, and is a successful algorithm on many problems.


| Previous                          |                         Next |
| :-------------------------------- | ---------------------------: |
| [[n-step reinforcement learning]] | [[Q-function approximation]] |
