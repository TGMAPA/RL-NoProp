# Deep Q-Network (DQN) — Algorithm Explained Step by Step

---

## Step 0. Initialize Everything

### Initialize the Online Q-Network
- This neural network approximates the function `Q(s, a)`.
- Its weights are initialized randomly.
- At this point, its predictions are meaningless.

### Initialize the Target Network
- Create a second network with the same architecture as the online network.
- Copy the weights from the online network.
- This network will remain fixed for a while to stabilize learning.

### Initialize the Replay Buffer
- Create a memory to store past experiences.
- This buffer will hold tuples `(s, a, r, s', done)`.

### Initialize ε (Exploration Rate)
- Start with a high value (e.g. `ε = 1.0`) to encourage exploration.
- Plan a schedule to slowly decrease ε over time.

---

## Step 1. Observe the Initial State
- Reset the environment.
- Obtain the initial state `s`.
- No learning happens yet.

---

## Step 2. Select an Action (ε-greedy)

For the current state `s`:

### With probability ε
- Select a random action.
- This ensures exploration.

### With probability 1 − ε
- Use the online Q-network to compute `Q(s, a)` for all actions.
- Select the action with the highest predicted Q-value: `a = argmaxₐ Q(s, a)`


- This is exploitation.

> Even early in training, this step is used.  
> Exploration dominates because ε is high.

---

## Step 3. Execute the Action
- Apply the selected action `a` in the environment.
- Observe:
  - the reward `r`
  - the next state `s'`
  - whether the episode has terminated (`done`)

---

## Step 4. Store the Experience
- Save the transition `(s, a, r, s', done)` in the replay buffer.
- This experience may be reused many times during training.

---

## Step 5. Sample a Mini-Batch from the Replay Buffer
- Randomly sample a mini-batch of transitions from the buffer.
- These samples come from different times and situations.
- This breaks temporal correlations and stabilizes training.

> Learning is decoupled from acting.

---

## Step 6. Compute the Target Value for Each Transition

For each sampled transition `(s, a, r, s', done)`:

### If the episode terminated at `s'`

`y = r`

### Otherwise

`y = r + γ maxₐ Q_target(s', a)`


- The target network is used here.
- The result `y` is a scalar target value.

---

## Step 7. Compute the Loss
- Use the online network to compute the predicted value:

`Q(s, a)`

- Compute the loss (typically Mean Squared Error):

`L = (y − Q(s, a))²`


- This loss measures how wrong the current Q-estimate is.

---

## Step 8. Update the Online Network
- Perform gradient descent on the loss.
- Update the parameters of the online Q-network.
- The network becomes slightly better at predicting correct Q-values.

---

## Step 9. Update the Target Network (Periodically)
- Every `N` steps, copy the online network weights into the target network:

`θ⁻ ← θ`

- Between updates, the target network remains fixed.
- This stabilizes the learning target.

---

## Step 10. Decay ε
- Reduce ε according to a predefined schedule.
- Exploration decreases gradually.
- Exploitation becomes dominant as learning progresses.

---

## Step 11. Repeat
- Continue steps 2–10 until:
  - the task is solved, or
  - a maximum number of steps is reached.