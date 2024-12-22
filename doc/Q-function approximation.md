>[!abstract] Learning outcomes
>- Manually apply linear Q-function approximation to solve small-scale MDP problems given some known features.
>- Select suitable features and design & implement Q-function approximation for model-free reinforcement learning techniques to solve medium-scale MDP problems automatically.
>- Argue the strengths and weaknesses of function approximation approaches.
>- Compare and contrast linear Q-learning with deep Q-learning.

## Overview
Using a Q-table has two limitations:
- It requires visiting every reachable state many times and apply every action many times to get a good estimate of $Q(s,a)$. Thus, if a state $s$ is never visited, there is no estimate of $Q(s,a)$.
- It requires us to maintain a table of size $|A|\times|S|$, which is prohibitively large for any non-trivial problem.
$\to$ Use machine learning to approximate Q-functions. In particular, we will look at **linear function approximation** and approximation using **deep learning** (deep Q-learning). Instead of calculating an exact Q-function, we approximate it using simple methods that both eliminate the need for a large Q-table (therefore the methods scale better), and also allowing use to provide reasonable estimates of $Q(s,a)$ _even if we have not applied action a in state s previously_.
## Linear Q-learning (Linear Function Approximation)
The key idea is to **approximate** the Q-function using a linear combination of **features** and their **weights**. Instead of recording everything in detail, we think about what is most important to know, and model that. The overall process is:
- for the states, consider what are the features that determine its representation;
- during learning, perform updates based on the **weights of features** instead of states; and
- estimate $Q(s,a)$ by summing the features and their weights.
### Linear Q-function representation


