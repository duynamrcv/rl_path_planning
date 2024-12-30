>[!abstract] Learning outcomes
>- Define "extensive form game".
>- Identify situations in which extensive form games are a suitable model of a problem.
>- Define the types of strategy for an extensive form game.
>- Manually apply backward induction to small extensive form games.
>- Design and implement backward induction to solve medium-scale extensive form games automatically.

## Overview
There are three main ways to solve extensive form games:
1. In a model-based game, we can use **backward induction**, which is where we calculate an equilibrium for every subgame of the game, and these to decide our moves.
2. In a model-free game, we can use **model-free reinforcement learning**. These are very similar to techniques such as Q-learning or policy gradient methods, except that there are other players that can effect our rewards, rather than just ourselves and the environment.
3. If we have a simulation, we can use model-free techniques or **multi-agent Monte-Carlo tree search** (MCTS), which is similar to single-agent MCTS, except again there are other players that can effect our rewards, rather than just ourselves and the environment.
## Perfect information extensive form games
>[!info] Definition - Perfect information extensive form games
>A perfect information extensive form game is a tuple $G=(N,S,s_0,A,T,r)$
>- $N$ is a set of n **number of players**
>- $S$ is a set of **states** (or **nodes**)
>- $s_0$ is the **initial state**
>- $A:D\to2^A$ is a function that specifies the allowed actions from each state $s\in S$
>- $P:S\to N$ is a function that specifies which player chooses the action at a node (whose turn it is)
>- $T:S\times A\to S$ is a **transition function** that specifies the successor state for choosing an action in a state
>- $r:S\to\mathbb{R}^N$ is a **reward function** that returns an $N$-tuple specifying the payoff each player receives in state $S$.

The interface of an extensive form games:
```python
class ExtensiveFormGame:

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def get_players(self): abstract

    ''' Get the valid actions at a state '''
    def get_actions(self, state): abstract

    ''' Return the state resulting from playing an action in a state '''
    def get_transition(self, state, action): abstract

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def get_reward(self, state, action, next_state): abstract

    ''' Return true if and only if state is a terminal state of this game '''
    def is_terminal(self, state): abstract

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def get_player_turn(self, state): abstract

    ''' Return the initial state of this game '''
    def get_initial_state(self): abstract

    ''' Return a game tree for this game '''
    def game_tree(self):
        return self.state_to_node(self.get_initial_state())

    def state_to_node(self, state):
        if self.is_terminal(state):
            node = GameNode(state, None, self.get_reward(state))
            return node

        player = self.get_player_turn(state)
        children = dict()
        for action in self.get_actions(state):
            next_state = self.get_transition(state, action)
            child = self.state_to_node(next_state)
            children[action] = child
        node = GameNode(state, player, None, children = children)
        return node

class GameNode:

    # record a unique node id to distinguish duplicated states
    next_node_id = 0

    def __init__(self, state, player_turn, value, is_best_action = False, children = dict()):
        self.state = state
        self.player_turn = player_turn
        self.value = value
        self.is_best_action = is_best_action
        self.children = children

        self.id = GameNode.next_node_id
        GameNode.next_node_id += 1
```
## Solutions for extensive form games
>[!info] Definition – Pure strategy
A **pure strategy** for an extensive form game $G$, the pure strategies for a player $i$ is the Cartesian product $\prod_{s\in S,P(s)=i}A(s)$.

Therefore, a pure strategy for player $i$ tells them what move to take in each state where it is their turn. An optimal solution for an extensive form game is called the **subgame-perfect equilibrium** for that game. Before defining what subgame-perfect equilibria are, first defining what sub-games are.
>[!info] Definition - Sub-game
>Given an extensive form game $G$, the **sub-game** of $G$ rooted at the node $s_g\in S$ is the game $G_{s_g}=(N,S,s_g,A,T,r)$; that is, the part of the game tree in which $s_g$ is the root node and its descendants are the same as its descendants in $G$.

>[!info] Definition – Subgame-perfect equilibria
The **subgame-perfect equilibria** (SPE) of a game $G$ consists of all strategy profiles for the $N$ agents such that for any subgame $G_{s_g}$ of $G$, the strategy for player $P(s_g)$ is the best response for that player at $s_g$.

Therefore, a sub-game perfect equilibria is the best response for every agent in the game when it is their turn.

| Previous              |                   Next |
| :-------------------- | ---------------------: |
| [[Normal form games]] | [[Backward induction]] 
