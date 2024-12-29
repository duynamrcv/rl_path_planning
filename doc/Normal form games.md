The term **"game"** is a more general term to describe a problem that involves **multiple agents or players**.
The standard definition of an MDP is for a single agent, who controls all of the actions. In a multi-agent system, further challenges: the effects of actions and the rewards are dependent also on the actions of the agents. In fact, the other agents can even be our adversaries, so they may be working to minimize rewards.
First, **normal form games** are single-shot (non-sequential) games where a group of agents each has to play a move (execute an action) at the same time as all others, but the reward (or **payoff** that they receive is dependent on the moves of the other agents. Then **extensive form games** are sequential games, meaning that there are multiple actions played in sequence.
>[!abstract] Learning outcomes
>- Identify situations in which normal form games are a suitable model of a problem.
>- Manually calculate the best responses and Nash equilibria for two-player normal form games.
>- Compare and contrast pure strategies and mixed strategies.

## Overview
Normal form games capture many different applications in the field of multi-agent systems.
>[!info] Definition - Normal form game
>A **normal form game** is a tuple $G=(N,A,u)$
>- $N$ is a set of $n$ number players
>- $A=A_1\times...\times A_n$ is an **action profile**, where $A_i$ is the set of actions for player $i$. Thus, an action profile $a=(a_1,...,a_n)$ describes the simultaneous moves by all players.
>- $u:A\to\mathbb{R}^N$ is a reward function that returns an $N$-tuple specifying the payoff each player receives in state $S$. This is called the **utility** for an action.

Normal game games can be visualized as matrices, with each agent representing one dimension of the matrix, each row represents an action for a player, and each cell represents the utility received when the players each take the action.
## Solutions for normal form games: strategies
In normal form games, the solution for a player in the game is known as a **strategy**. There are several types of strategy.
>[!info] Definition - Pure strategy
>A **pure strategy** for an agent is when the agent selects a single action and plays it. If the agent were to play the game multiple times, they would choose the same action every time.

>[!info] Definition - Mixed strategy
>A **mixed strategy** is when an agent selects the action to play based on some probability distribution. That is, we choose action a with probability 0.8 and action b with probability 0.2. If the agent were to play the game an infinite number of times, it would select a 80% of the time and b 20% of the time.

Note that a pure strategy is a special case of a mixed strategy where one of the actions has a probability of 1. The set of strategies for an agent is called a **strategy profile**. The strategy profile for agent $i$ is denoted $S_i$. Note that $S_i\neq A_i$ because $S_i$ contains mixed strategies. The set of mixed-strategy profiles for all agents is simply $S=S_1\times...\times S_n$. The notation $S_{-i}$ to denote the set of mixed-strategy profiles for all agents except for agent $i$, and $s_{-i}\in S_{-i}$ to denote an element of this.
>[!info] Definition - Dominant strategy
>Strategy $s_i$ for player $i$ **weakly dominates** strategy $s'_i$ if the utility received by the agent for playing strategy $s_i$ is greater than or equal to the utility received by that agent for playing $s'_i$. Formally, $s_i$ weakly dominates $s'_i$ if and only if:
>$$\forall s_{-i}\in S_{-i},u_{i}(s_{i},s_{-i})\geq u_{i}(s'_{i},s_{-i})$$
>Strategy $s_i$ **strongly dominates** strategy $s'_i$ if its utility is strictly greater. Formally:
>$$\forall s_{-i}\in S_{-i},u_{i}(s_{i},s_{-i})>u_{i}(s'_{i},s_{-i})$$
>A strategy is a **weakly (resp. strictly) dominant strategy** if it weakly (resp. strictly) dominates all other strategies.

## Best response and Nash equilibria
How to solve games from the perspective of one of the agents, known as the agent’s **best response**. Then, the concept of **equilibria**, which captures solutions to the entire game. Informally, the concept of a best response refers to the best strategy that an agent could select _if_ it know how all of the other agents in the game were going to play.
>[!info] Definition - Best response
>The **best response** for an agent $i$ if its opponents play strategy profile $s_{-i}\in S_{-i}$ is a mixed strategy $s_i^*\in S_i$ such that $u(s_i^*,s_{-i})\geq u(s'_i,s_{-i})$ for all strategies $s'_i\in S_i$.

Note that for many problems, there can be multiple best responses.
Nash equilibrium is a **stable** strategy profile for all agents in $N$ such that no agent has an incentive to change strategy if all other agents kept their strategy the same.
>[!info] Definition - Nash equilibria
>A strategy profile $s=(s_1,...,s_n)$ is a **Nash equilibrium** if for all agents $i$ and for all strategies $s_i$ is a best response to the strategy $s_{-i}$.

If the strategies in a Nash equilibrium are all pure strategies, then we call this a **pure-strategy Nash equilibrium**. Otherwise, if is a **mixed-strategy Nash equilibrium**.
## Calculating best response and Nash equilibria
The set of best responses for a normal form game can be calculated by searching through all strategies to find those with the highest payoff.
>[!todo] Algorithm 1 - Best response
>**Input**: Normal form game $G=(N,A,u)$, agent $i$, and strategy profile $s_{-i}$ for agents other than $i$
>**Output**: Set of best responses
>best_response $=\varnothing$
>best_response_value $=-\infty$
>**for each** $s_i\in S_i$
>$\qquad$**if** $u(s_i,s_{-i})>$ best_response_value **then**
>$\qquad\qquad$best_response $=\{s_i\}$
>$\qquad$**else if** $u(s_i,s_{-i})=$ best_response_value **then**
>$\qquad\qquad$best_response $=$ best_response $\cup\{s_i\}$
>**return** best_response

This has complexity $O(|Si|)$.
>[!todo] Algorithm 2 - Nash equilibria
>**Input**: Normal form game $G=(N,A,u)$
>**Output**: Set of Nash equilibria
>nash_equilibria $=\varnothing$
>**for each** $s_1\in A_1$
>$\qquad$**for each** $s_2 \in A_2$
>$\qquad\qquad$**if** $s_i\in BestResponse(i,s_j)$ **and** $s_j\in BestResponse(j,s_i)$ **then**
>$\qquad\qquad\qquad$ nash_equilibria $=$ nash_equilibria $\cup \{(s_i,s_j)\}$
>**return** nash_equilibria

## Mixed strategies
>[!info] Definition – Expected utility of a pure strategy
>**Expected utility** is the weighted average received by an playing a particular pure strategy. For an action $a_i\in A_i$, the expected utility of that action is:
>$$U_i(a_i)=p_1\times u_i(a_i,a_{-i1})+...+p_m\times u_i(a_i,a_{-im})$$
>where $a{-i1},...,a_{-im}$ are the action profiles for all agents other than $i$, and $p_1,...,p_m$ are the probabilities of our opponents playing those action profiles.

>[!info] Definition – Indifference
>An agent $i$ is **indifferent** between a set of pure strategies $I\subseteq A_i$ if for all $a_i,a_j\in I$, we have that $U_i(a_i)=U_i(a_j)$.

> [!info] Definition – Mixed-strategy Nash equilibria
> A **mixed-strategy Nash equilibria** is a mixed-strategy profile $S$ such that the strategy for each agent $i\in N$ is a tuple of probabilities $P_i=(p_1,...,p_m)$, one for each pure strategy, such that $p_1+...+p_m=1$ and that all opponents $j$ are indifferent to their pure strategies $A_j$.

## Takeaways
>[!success] Takeaways
>- **Normal form games** model non-sequential games where agents take actions simultaneously.
>- **Pure strategies** and **mixed strategies** can be used by agents. We analyze which strategies are good and also how these relate to Nash equilibria.

| Previous                 |                     Next |
| :----------------------- | -----------------------: |
| [[Actor-critic methods]] | [[Extensive form games]] |
