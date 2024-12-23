**Policy-based** methods for reinforcement learning are introduced. Policy-based methods learn a policy directly, rather than learning the value of states and actions.
Learning a policy directly has advantages, particularly for applications where the state space or the action space are massive or infinite. If the action space is infinite, then we cannot do [[Markov Decision Processes#Policy extraction|policy extraction]] because that requires us to iterate over all actions and extract the one that maximizes the reward. Learning the policy directly mitigates this.
In this chapter, we will introduce two policy-based methods:
- [**Policy iteration**](https://gibberblot.github.io/rl-notes/single-agent/policy-iteration.html#sec-policy-iteration): Like value iteration, this is a dynamic programming-based method for model-based MDPs.
- [**Policy gradients**](https://gibberblot.github.io/rl-notes/single-agent/policy-gradients.html#sec-policy-based-policy-gradients): This is a model-free technique that performs gradient ascent, which is the same as the well-known gradient descent technique, but which maximizes rewards instead of minimizing error.

| Previous                     |                 Next |
| :--------------------------- | -------------------: |
| [[Q-function approximation]] | [[Policy iteration]] |
