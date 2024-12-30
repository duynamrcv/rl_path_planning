>[!abstract] Learning outcomes
>- Manually apply backward induction to solve small-scale extensive form games.
>- Design and implement a backward induction algorithm to solve medium-scale extensive form games automatically.

## Overview
**Backward induction** is a model-based technique for solving extensive form games. It solves this by recursively calculating the sub-game equilibrium for each sub-game and then using this to solve the parent node of each subgame. Because it solves subgames first, it is effectively solving the game backwards.
The intuition is as follows: starting at the terminal nodes of the tree (those were $A(s)$ is empty), for the parent, calculate the best move for the agent whose turn it is. This gives us the sub-game equilibrium for the smallest sub-games in the game. As the solutions are reward tuples themselves, we can solve the parent of the parents by using the parent solution as reward for the sub-game, which gives us the sub-game equilibrium for the game that starts at the parents of the parents of the terminal nodes. We progressively induct these values backward up the tree until we reach the start node.
## Algorithm
In the following algorithm, $best\_child$ is an $N$-tuple that is used to find the best child value for whose turn it is; while $best\_child(P(s))$ and $child\_reward(P(s))$ return the value from the $best\_child$ and $child\_reward$ tuple for the player who is choosing the action from state $s$.
>[!todo] Algorithm 1 - Backward induction
>**Input**: Extensive form game $G=(N,S,s_0,A,T,r)$
>**Output**: Sub-game equilibrium for each state $s\in S$
>**return** $BackwardInduction(s_0)$
>**function** $BackwardInduction(s\in S)$
>$\qquad$**if** $A(s)=\varnothing$ **then**
>$\qquad\qquad$**return** $r(s)$
>$\qquad best\_child\leftarrow(-\infty,...,-\infty)$
>$\qquad$**for each** $a\in A(s)$
>$\qquad\qquad s'\leftarrow T(s,a)$
>$\qquad\qquad child\_reward\leftarrow BackwardInduction(s')$
>$\qquad\qquad$**if** $child\_reward(P(s))>best\_child(P(s))$ **then**
>$\qquad\qquad\qquad best\_child=child\_reward$
>$\qquad$**return** $best\_child$

The solution above is a recursive algorithm that returns the reward tuple for a terminal node, and otherwise finds the best reward tuple for the children of the node.
## Implementation
```python
from extensive_form_game import GameNode

class BackwardInduction:
    def __init__(self, game, do_cache = False):
        self.game = game
        self.do_cache = do_cache
        self.cache = dict()

    def backward_induction(self, state):

        if self.game.is_terminal(state):
            node = GameNode(state, None, self.game.get_reward(state))
            return node

        best_child = None
        best_action = None
        player = self.game.get_player_turn(state)
        children = dict()
        for action in self.game.get_actions(state):
            next_state = self.game.get_transition(state, action)
            child = self.backward_induction(next_state)
            if best_child is None or child.value[player] > best_child.value[player]:
                if best_child is not None:
                    best_child.is_best_action = False
                child.is_best_action = True
                best_child = child
            children[action] = child
        node = GameNode(state, player, best_child.value, children = children)
        return node

    def backward_induction_with_cache(self, state):

        state_key = self.game.to_string(state)
        if self.do_cache and state_key in self.cache.keys():
            return self.cache[state_key]

        if self.game.is_terminal(state):
            node = GameNode(state, None, self.game.get_reward(state))
            if self.do_cache:
                self.cache[state_key] = node
            return node

        best_child = None
        best_action = None
        player = self.game.get_player_turn(state)
        children = dict()
        for action in self.game.get_actions(state):
            next_state = self.game.get_transition(state, action)
            child = self.backward_induction(next_state)
            if best_child is None or child.value[player] > best_child.value[player]:
                if best_child is not None:
                    best_child.is_best_action = False
                child.is_best_action = True
                best_child = child
            children[action] = child
        node = GameNode(state, player, best_child.value, children = children)
        if self.do_cache:
            self.cache[state_key] = node
        return node
```

| Previous                 |                                   Next |
| :----------------------- | -------------------------------------: |
| [[Extensive form games]] | [[Multi-agent Reinforcement Learning]] |
