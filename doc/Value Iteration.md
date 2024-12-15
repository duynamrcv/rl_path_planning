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
> **until** $\delta\leq\theta$

The method is just applying the Bellman equation until either the value function $V$ doesn’t change anymore, or until it changes in by a very small amount $\theta$.  Using the idea of Q-values:
> [!todo] Algorithm 2 - Q-values iteration
> **Input**: MDP $M=\left\langle S,s_0,A,P_a(s'|s),r(s,a,s')\right\rangle$
> **Output**: Value function $V$
> Set V to arbitrary value function; e.g., $V(s)=0$ for all $s$
> **repeat**
> $\qquad\delta\leftarrow0$
> $\qquad$**foreach** $s\in S$
> $\qquad\qquad$**foreach** $a\in A(s)$
> $\qquad\qquad\qquad Q(s,a)\leftarrow\sum_{s'\in S}P_a(s'|s)\left[r(s,a,s')+\gamma V(s')\right]$
> $\qquad\qquad\delta\leftarrow\max(\delta,\left\vert \max_{a\in A(s)}Q(s,a)-V(s)\right\vert)$
> $\qquad V\leftarrow \max_{a\in A(s)}Q(s,a)$
> **until** $\delta\leq\theta$

Value iteration converges to the optimal policy as iterations continue: $V\to V^∗$ as $i\to\infty$, where $i$ is the number of iterations. Value iteration  **converges** to the optimal value function $V^*$ asymptotically, but practically, he algorithm terminates when the **residual** $\Delta$ reaches some pre-determined threshold $\theta$ – that is, when the largest change in the values between iterations is "small enough".
A policy can now be easily defined: in a state s, given V, choose the action with the highest expected reward using [[Markov Decision Processes#Policy extraction|policy extraction]].
The loss of the result greedy policy terminating after k iterations is bounded by $\dfrac{2\gamma\delta_\text{max}}{1-\gamma}$, where $\delta_\text{max}=\max_s{\left\vert V^*(s)-V_k(s)\right\vert}$.
> [!note] Note
> **Stochastic Value Policy** uses a [[Multi-armed bandits]] to sometimes select a random action instead of the action that leads to the best value. Why?
> Because a policy can contain the loop, for example:
> $$...s_{1}\stackrel{a_{1}}{\to}s_{2}\stackrel{a_{2}}{\to}s_{1}\stackrel{a_{1}}{\to}s_{2}\stackrel{a_{2}}{\to}...$$
> If a policy contains a loop, it will loop infinitely when:
> - the policy is deterministic (not stochastic);
> - the environment actions are deterministic (not stochastic).
>  By sometimes selecting a random action, we will eventually exit the loop.

## Complexity
The complexity of each iteration is $O(|S|^2|A|)$. On each iteration, we need iterate over all state $\left(\sum_{s'\in S}\right)$, meaning $|S|^2$ iterations.  But also within each outer loop iteration, we need to calculate the value for every action to find the maximum. The number of iterations depends on the value of $\theta$ and can not be predefined.
## Takeaways
> [!success] Takeaways
> - Value iteration is an algorithm for calculating a value function $V$, from which a policy can be extracted using policy extraction.
> - It produces an optimal policy an infinite amount of time.
> - For medium-scale problems, it can converge on the optimal policy in a "reasonable" amount of time, but does not scale as well as some other techniques.

| Previous                |                    Next |
| :---------------------- | ----------------------: |
| [[Value-based methods]] | [[Multi-armed bandits]] |
