from DenoisingMLP import DenoisingMLP
from BanditsEnv import BanditsEnv
from Diffusion import Diffusion
from TrainingLoop import train
import torch, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -- Config
# SLOTS = [
#     {"outcomes": [0, 10], "probabilities": [0.6, 0.4]},
#     {"outcomes": [0, 100], "probabilities": [0.97, 0.03]},
# ]

SLOTS = [
    {"outcomes": [0, 100], "probabilities": [0.2, 0.8]},
    {"outcomes": [0, 100], "probabilities": [0.8, 0.2]},
]

LR = 1e-3

EPISODES = 5000

# Diffusion hyperparameters

T = 5

alpha = torch.linspace(1.0, 0.1, T)


# -- Initialize environment
ENV = BanditsEnv(SLOTS)

# -- NoProp MLPs
mlps = nn.ModuleList([DenoisingMLP(ENV.num_actions) for _ in range(T)])
optimizers = [optim.Adam(mlp.parameters(), lr=LR) for mlp in mlps]

# -- Initialize Epsilon-greedy
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# Diffusion
diffuser = Diffusion(T, alpha)


# Execute training loop
q_history = train(
        ENV,
        EPISODES,
        mlps,
        optimizers,
        diffuser,
        epsilon,
        epsilon_min,
        epsilon_decay,
    )


#  Graphs
q_history = np.array(q_history)

plt.figure(figsize=(8, 5))
for i in range(q_history.shape[1]):
    plt.plot(q_history[:, i], label=f"Q(slot {i})")

plt.xlabel("Training checkpoints")
plt.ylabel("Q-value")
plt.title(f"NoProp - DQN learning on bandit")
plt.legend()
plt.tight_layout()
plt.show()



# Test

def greedy_policy(state):
    with torch.no_grad():
        z = torch.randn(1, ENV.num_actions, device=state.device)

        for t in reversed(range(diffuser.T)):
            z = mlps[t](state, z)

        return z.argmax(dim=1).item()

def random_policy(_):
    return random.randint(0, ENV.num_actions - 1)

def evaluate(policy_fn, steps=5000):
    total_reward = 0.0
    state = torch.tensor([[1.0]], dtype=torch.float32)

    for _ in range(steps):
        action = policy_fn(state)
        reward = ENV.step(action)
        total_reward += reward

    return total_reward / steps

avg_learned = evaluate(greedy_policy)
avg_random = evaluate(random_policy)

print(f"Average reward (learned): {avg_learned:.3f}")
print(f"Average reward (random):  {avg_random:.3f}")
