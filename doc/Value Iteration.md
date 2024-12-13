> [!abstract] Learning outcomes
> - Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to solve medium-scale MDP problems automatically
> - Construct a policy from a value function
> - Discuss the strengths and weaknesses of value iteration

## Overview
**Value iteration** is a dynamic-programming method to finding the optimal value function $V^*$ by solving the Bellman equations iteratively, improving $V$ until it converges to $V^*$.
## Algorithm
We just repeatedly calculate $V$ using the Bellman equation until converge to the solution or execute a pre-determined number of iterations.
> [!todo] Algorithm 1 - Value iteration
> **Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
> **Output**: Value function $V$
> Set V to arbitrary value function; e.g., $V(s)=0$ for all $s$
> **repeat**
> $\qquad\delta\leftarrow0$
> $\qquad$**foreach** $s\in S$
> $\qquad\qquad$ Compute Bellman equation $V'(s)\leftarrow\max_{a\in A(s)}\sum_{s'\in S}P_a(s'|s)\left[r(s,a,s')+\gamma V(s')\right]$
> $\qquad\qquad\delta\leftarrow\max(\delta,\left\vert V'(s)-V(s)\right\vert)$
> $\qquad V\leftarrow V'$
> **until** $\delta\leq0$
