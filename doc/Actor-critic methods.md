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
