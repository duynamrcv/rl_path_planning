## What is reinforcement learning?
Reinforcement learning (RL) is *learning from experience*.
RL is a branch of Machine learning, in which, **agents** learn to make **sequential decisions** in environments., guided by a set of rewards and penalties.
Difference from supervised machine learning:
- **Experience instead of labels**: 
	- In supervised machine learning, we need inputs and labelled outputs
	- In reinforcement learning, 'labels'  are provided by environments, called **rewards**.
	- Rewards can be **positive** or **negative**.
	- The aim is to learn behavior that increases the cumulative reward over a number of decisions.
- **Decisions are sequential**:
	- Reinforcement learning is **sequential decision making**.
	- Feedback is reward, and is received at each step.
## Single-agent reinforcement learning
[[Markov Decision Processes]] (MDPs) allow us to model reinforcement learning problems.
- *model-based* techniques, where the entire MDP model is known
- *model-free* techniques, which are flexible enough that when some information is not provide explicitly, but can be sampled enough time, it still learn good behavior.
[[Value-based methods]]
[[Policy-based methods]]
[[Modeling and abstraction for MDPs]]
## Multi-agent reinforcement learning
*Multi-agent MDPs* (a.k.a *games*), in which there are multiple agents in a problem, and need to plan decisions while considering other actors in the environment. Both consider *model-based* and *model-free* techniques.
[[Normal form games]]
[[Extensive form games]]
[[Modelling and abstraction for multi-agent games]]

|                          Next |
| ----------------------------: |
| [[Markov Decision Processes]] |
