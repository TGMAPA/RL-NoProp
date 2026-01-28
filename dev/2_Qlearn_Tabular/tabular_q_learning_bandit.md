# Tabular Q-Learning for a Multi-Armed Bandit (Slot Machines)

This document explains **how** and **why** tabular Q-learning applies to the multi-armed bandit problem (slot machines).

It demonstrates how the standard Q-learning update rule simplifies under the special conditions of a bandit problem and how this simplified rule is used to learn the expected reward of each action.

## 1. Problem Overview: The Multi-Armed Bandit

A multi-armed bandit is a reinforcement learning problem defined by the following characteristics:

- Each slot machine corresponds to one **action**
- Pulling a machine produces a **stochastic** (random) reward
- Rewards are drawn from a **fixed but unknown** probability distribution
- Actions are **statistically independent** — past choices do not affect future reward distributions

There are **no states** and **no state transitions**.

### Objective
Identify the slot machine with the highest **expected reward** and select it as frequently as possible over time, while still performing sufficient exploration to ensure high-confidence estimates.

## 2. What Standard Q-Learning Solves

In general reinforcement learning, Q-learning estimates a function:

$$ Q(s, a) $$

This value represents the **expected cumulative discounted return** obtained by:

- starting in state $s$
- taking action $a$
- and then following an optimal policy thereafter

In other words, $Q(s, a)$ answers:  
“How good is it to take action $a$ in state $s$, taking into account both the immediate reward and all optimally expected future rewards?”

## 3. The General Q-Learning Update Rule

The standard off-policy Q-learning update is:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr] $$

This update consists of three conceptual components:

1. **Current estimate**  
   $Q(s, a)$  
   The agent’s present belief about the value of this state–action pair.

2. **Target value**  
   $r + \gamma \max_{a'} Q(s', a')$  
   - $r$ : immediate reward received  
   - $\gamma \max_{a'} Q(s', a')$ : discounted value of the best action in the resulting state

3. **Temporal-difference error** (correction term)  
   $\Bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr]$  
   The difference between what was observed/predicted and what was expected.

The learning rate $\alpha \in (0,1]$ controls the weight given to the new information.

## 4. Why the Bandit Problem Is a Special Case

A multi-armed bandit violates several assumptions of standard Q-learning:

- There is only **one global state** (or the state is completely uninformative)
- The environment is **stationary** and memoryless
- Actions do **not influence** future situations
- Rewards are **immediate** — there is no delayed consequence

Consequently:

- The state variable $s$ carries **no useful information**
- There is **no meaningful next state** $s'$
- There is **no sequence of future rewards** to plan over

## 5. Reduction: $Q(s, a) \rightarrow Q(a)$

Because the state is constant and uninformative, we can safely remove it from the notation:

$$ Q(s, a) \quad \rightarrow \quad Q(a) $$

Now $Q(a)$ directly represents:

**The expected reward** obtained by selecting action $a$ (i.e., pulling a particular slot machine).

These values can be stored in a simple one-dimensional table (a vector of length equal to the number of arms).

## 6. Elimination of the Future-Value Term

In the general update, the term

$$ \gamma \max_{a'} Q(s', a') $$

estimates future discounted returns.

In a pure bandit problem:

- there is **no next state** $s'$
- rewards do **not accumulate** over episodes
- actions have **no delayed effects**

Therefore, we can (and usually do) set:

$$ \gamma = 0 $$

This completely removes the future term from the update.

## 7. Simplified Q-Learning Update for Bandits

Applying the bandit assumptions to the general rule:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr] $$

yields:

$$ Q(a) \leftarrow Q(a) + \alpha \Bigl[ r - Q(a) \Bigr] $$

This is the well-known **incremental sample-average update** (a form of stochastic gradient descent on squared error).

## 8. Practical Usage of the Update Rule

On each time step the agent performs the following loop:

1. Select an action $a$ (using an exploration strategy: $\varepsilon$-greedy, UCB, Thompson sampling, etc.)
2. Pull the corresponding slot machine and observe reward $r$
3. Update **only** the value of the selected action:

$$ Q(a) \leftarrow Q(a) + \alpha \Bigl[ r - Q(a) \Bigr] $$

This update rule has a very intuitive effect:

- If $r > Q(a)$ → the estimate **increases** (the action was better than expected)
- If $r < Q(a)$ → the estimate **decreases** (the action was worse than expected)

Over many samples, $Q(a)$ converges (in expectation) to the true expected reward $\mathbb{E}[R \mid a]$.

## 9. Key Insight

Tabular Q-learning **remains valid** for multi-armed bandit problems, but the absence of states and future consequences dramatically simplifies the algorithm.

What remains is **pure expected-reward estimation** for each independent action through repeated sampling and incremental averaging.