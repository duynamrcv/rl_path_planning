> [!abstract] Learning outcomes
> - Manually apply n-step reinforcement learning approximation to solve small-scale MDP problems.
> - Design and implement n-step reinforcement learning to solve medium-scale MDP problems automatically.
> - Argue the strengths and weaknesses of n-step reinforcement learning.

## Overview
Some weaknesses of Q-learning and SARSA:
1. Unlike Monte-Carlo methods, which reach a reward and then backpropagate this reward, TD methods use bootstrapping (estimate future discounted reward using $Q(s,a$)). It can take a long time to for rewards to propagate throughout a Q-function.
2. Rewards can be sparse, meaning that there are few state/actions that can lead to non-zero rewards. This is problems because, reinforcement learning algorithms behave entirely randomly and will struggle to find good rewards.
3. Both methods estimate a Q-function $Q(s,a)$, and the simplest way to model this is Q table. However,  this requires to maintain a table of size $|A|\times|S|$, which is large for any non-trivial problem.
4. Using a Q-table requires visiting every reachable state many times and applying every action many times to get a good estimate of $Q(s,a)$. Thus, if we never visit a state $s$, we have no estimate of $Q(s,a)$.
To get around limitations 1 and 2, using **n-step temporal difference learning**: 'Monte-Carlo' techniques executing entire episodes and then backpropagate the reward, while basic TD methods only look at the reward in the next step, estimating future rewards. *n-step* methods instead look $n$ steps ahead for the reward before updating the reward, and then estimate the remainder.
![[n-step.png]]
n-step TD learning comes from the idea in the image above. Monte-Carlo methods uses 'deep backups', where entire episodes are executed and the reward backpropagated. Methods such as Q-learning and SARSA use 'shallow backups', only using the reward from the 1-step ahead. n-step learning finds the middle ground: only update the Q-function after having explored ahead $n$ steps.
## n-step TD learning
$n$ is the parameter that determines the number of steps that using to look ahead before updating the Q-function. For $n=1$, this is just 'normal' TD learning. When $n=2$, the algorithm looks one step beyond the immediate reward, $n=3$ it look two steps beyond,... Both Q-learning and SARSA have an n-step version.
### Intuition
The details and algorithm for n-step reinforcement learning make it seem more complicated than it really is. At an intuitive level, it is quite straightforward: at each step, instead of updating our Q-function or policy based on the reward received from the previous action, plus the discounted future rewards, we update it based on the last n rewards received, plus the discounted future rewards from n states ahead.
### Discounted future rewards
When calculating a discounted reward over a episode, simply suming up the rewards over the episode:
$$G_t   =   r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4 + \ldots$$
Re-write:
$$G_t =   r_1 + \gamma(r_2 + \gamma(r_3 + \gamma(r_4 + \ldots)))$$
If $G_t$ is the value received at time-step $t$, then:
$$G_t=r_t+\gamma G_{t+1}$$
In 1-step TD methods we do not know $G_{t+1}$ when updating $Q(s,a)$, so estimating using bootstrapping:
$$G_t=r_t+\gamma V(s_{t+1})$$
That is, the reward of the entire future from step $t$ is estimated as the reward at $t$ plus the estimated (discounted) future reward from $t+1$. $V(s_{t+1})$ is estimated using the maximum expected return (Q-learning) or the estimated value of the next action (SARSA). This is a **one-step return**.
### Truncated discounted rewards
Estimating **n-step returns**:
$$G^n_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots  \gamma^n V(s_{t+n})$$
In this above expression $G_t^n$ is the full reward, **truncated** at $n$ steps, at time $t$. The basic idea of n-step reinforcement learning is that we do not update the Q-value immediately after executing an action: we wait $n$ steps and update it based on the n-step return.
If $T$ is the termination step and $t+n\geq T$, then we just use the full reward.
In Monte-Carlo methods, we go all the way to the end of an episode. Monte-Carlo Tree Search is one such Monte-Carlo method, but there are others that we do not cover.
### Updating the Q-function
The update rule is then different. First, we need to calculate the truncated reward for $n$ steps, in which $\tau$ is the time step that we are updating for (that is, $\tau$ is the action taken $n$ steps ago):
$$G \leftarrow \sum^{\min(\tau+n, T)}_{i=\tau+1}\gamma^{i-\tau-1}r_i$$
This just sums the discounted rewards from time step $\tau+1$ until either $n$ steps $(\tau+n)$ or termination of the episode $(T)$, whichever comes first. Then calculate the n-step expected reward:
$$\text{If } \tau+n < T \text{ then } G \leftarrow G + \gamma^n Q(s_{\tau+n}, a_{\tau+n}).$$
This adds the future expect reward if we are not at the end of the episode (if $\tau+n<T$). Finally, we update the Q-value:
$$Q(s_{\tau}, a_{\tau}) \leftarrow  Q(s_{\tau}, a_{\tau}) + \alpha[G - Q(s_{\tau}, a_{\tau}) ]$$
In the update rule above, we are using a SARSA update, but a Q-learning update is similar.
### n-step SARSA
>[!todo] Algorithm 1 - n-step SARSA
>**Input**: MDP $M=\left\langle S,s_0,A,P-a(s'|s),r(s,a,s')\right\rangle$, number of steps $n$
>**Output**: Q-function $Q$
>Initialize $Q(s,a)=0$ for all $s$ and $a$
>**repeat**
>$\qquad$ Select action $a$ to apply in $s$ using Q-values in $Q$ and a multi-armed bandit algorithm
>$\qquad\vec{s} = \langle s\rangle$
>$\qquad\vec{a} = \langle a\rangle$
>$\qquad\vec{r} = \langle \rangle$
>$\qquad$**while** $\vec{s}$ is not empty **do**
>$\qquad\qquad$**if** $s$ is not a terminal state **then**
>$\qquad\qquad\qquad$Execute action $a$ in state $s$
>$\qquad\qquad\qquad$Observe reward $r$ and new state $s'$
>$\qquad\qquad\qquad$**if** $s'$ is not a terminal state **then**
>$\qquad\qquad\qquad\qquad$Select action $a'$ to apply in $s'$ using $Q$ and a multi-armed bandit algorithm
>$\qquad\qquad\qquad\qquad\vec{s}\leftarrow\vec{s}+\langle s'\rangle$
>$\qquad\qquad\qquad\qquad\vec{a}\leftarrow\vec{a}+\langle a'\rangle$
>$\qquad\qquad$**if** $|\vec{r}|=n$ **or** $s$ is a terminal state **then**
>$\qquad\qquad\qquad G\leftarrow\sum_{i=0}^{|\vec{r}|-1}\gamma^i\vec{r}^i$
>$\qquad\qquad\qquad$**if**$s$ is not a terminal state **then**
>$\qquad\qquad\qquad\qquad G\leftarrow G+\gamma^nQ(s',a')$
>$\qquad\qquad\qquad Q(\vec{s}_0,\vec{a}_0)\leftarrow Q(\vec{s}_0,\vec{a}_0)+\alpha[G-Q(\vec{s}_0,\vec{a}_0)]$
>$\qquad\qquad\qquad \vec{r}\leftarrow\vec{r}_{[1:n+1]}$
>$\qquad\qquad\qquad \vec{s}\leftarrow\vec{s}_{[1:n+1]}$
>$\qquad\qquad\qquad \vec{a}\leftarrow\vec{a}_{[1:n+1]}$
>$\qquad s\leftarrow s'$
>$\qquad a\leftarrow a'$
>**until** $Q$ converges

## Values of *n*
**Can we just increase $n$ to be infinity so that we get the reward for the entire episode?** Doing this is the same as [[Temporal difference reinforcement learning#Monte-Carlo reinforcement learning|Monte-Carlo reinforcement learning]], as we would no longer use TD estimates in the update rule. 
**What is the best value for $n$ then?** Unfortunately, there is no theoretically best value for n. It depends on the particular application and reward function that is being trained. In practice, it seems that values of n around 4-8 give good updates because we can easily assign credit to each of the 4-8 actions; that is, we can tell whether the 4-8 actions in the lookahead contributed to the score, because we use the TD estimates.
## Takeaways
>[!success] Takeaways
>- n-step reinforcement learning **propagates** rewards back $n$ steps to help with learning
>- It is simple, but the implementation requires a bot of 'book-keeping'.
>- Choosing a value of $n$ for a domain requires experimentation an intuition.


| Previous                                       |                        Next |
| :--------------------------------------------- | --------------------------: |
| [[Temporal difference reinforcement learning]] | [[Monte-Carlo Tree Search]] |
