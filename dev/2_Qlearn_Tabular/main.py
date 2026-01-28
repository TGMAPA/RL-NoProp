import random
import math
import matplotlib.pyplot as plt


# ===========================================================
# CONSTANTS
# ===========================================================
SLOTS = [
    {"outcomes": [0, 10], "probabilities": [0.6, 0.4]},
    {"outcomes": [0, 100], "probabilities": [0.97, 0.03]},
]

NUM_EPISODES = 10000
NUM_EVAL_STEPS = 5000

LEARNING_RATE = 0.01

EPSILON_MAX = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.001


# ===========================================================
# ENVIRONMENT
# ===========================================================
def pull_slot(slot_id):
    slot = SLOTS[slot_id]
    return random.choices(slot["outcomes"], slot["probabilities"])[0]


# ===========================================================
# Q-LEARNING (TRAINING)
# ===========================================================
def train_q_learning():
    num_slots = len(SLOTS)
    q_values = [0.0 for _ in range(num_slots)]
    q_history = [[] for _ in range(num_slots)]

    for episode in range(NUM_EPISODES):

        epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * math.exp(-EPSILON_DECAY * episode)

        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, num_slots - 1)
        else:
            action = q_values.index(max(q_values))

        reward = pull_slot(action)

        # Q-learning update
        q_values[action] += LEARNING_RATE * (reward - q_values[action])

        for i in range(num_slots):
            q_history[i].append(q_values[i])

    return q_values, q_history


# ===========================================================
# POLICY EVALUATION
# ===========================================================
def evaluate_policy(policy_fn, num_steps):
    total_reward = 0.0
    for _ in range(num_steps):
        action = policy_fn()
        total_reward += pull_slot(action)
    return total_reward / num_steps


# ===========================================================
# MAIN
# ===========================================================
q_values, q_history = train_q_learning()

# Learned policy (greedy)
def learned_policy():
    return q_values.index(max(q_values))

# Random baseline policy
def random_policy():
    return random.randint(0, len(SLOTS) - 1)

avg_reward_learned = evaluate_policy(learned_policy, NUM_EVAL_STEPS)
avg_reward_random = evaluate_policy(random_policy, NUM_EVAL_STEPS)

print("Learned Q-values:", q_values)
print(f"Average reward (learned policy): {avg_reward_learned:.3f}")
print(f"Average reward (random policy):  {avg_reward_random:.3f}")


# ===========================================================
# PLOT Q-VALUES ONLY
# ===========================================================
plt.figure(figsize=(8, 5))
for i in range(len(SLOTS)):
    plt.plot(q_history[i], label=f"Q(slot {i})")

plt.xlabel("Iterations")
plt.ylabel("Q-value")
plt.title("Tabular Q-learning – Slot Machines")
plt.legend()
plt.tight_layout()
plt.show()
