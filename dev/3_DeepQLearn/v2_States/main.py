from DQN import DQN
from BanditsEnv import BanditsEnv
from ReplayMemory import ReplayMemory
from TrainingLoop import train
import torch, random
import numpy as np
import matplotlib.pyplot as plt


# -- Config
BUFFER_CAPACITY = 2000

SLOTS = [
    {"outcomes": [10, 0], "probabilities": [0.8, 0.2]},
    {"outcomes": [0, 100], "probabilities": [0.2, 0.8]},
]

TARGET_UPDATE = 100

BATCH_SIZE = 64

LR = 1e-3

EPISODES = 100000



# -- Initialize environment
ENV = BanditsEnv(SLOTS)


# -- Initialize online and target network
# Trainable Net
onlineNet = DQN(8, ENV.num_actions)

# Target Net
targetNet = DQN(8, ENV.num_actions)

# Load trainable net parameters into target net 
targetNet.load_state_dict(onlineNet.state_dict())

# Set target net into evaluation mode 
targetNet.eval()

# Set optimizer
optimizer = torch.optim.Adam(onlineNet.parameters(), lr=LR)


# -- Initialize the Replay Buffer
buffer = ReplayMemory( capacity=BUFFER_CAPACITY )


# -- Initialize Epsilon-greedy
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995


# Execute training loop
q_history = train(
        ENV,
        onlineNet, 
        targetNet,
        buffer,
        EPISODES,
        optimizer,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size = BATCH_SIZE,
        targetNet_update = TARGET_UPDATE
    )



# ------ Test

def greedy_policy(state):
    with torch.no_grad():
        q_values = onlineNet(state)
        return torch.argmax(q_values).item()

def random_policy(_):
    return random.randint(0, ENV.num_actions - 1)

def evaluate(env, policy_fn, steps):
    total_reward = 0.0
    
    for _ in range(steps):
        state = env.getState()
        action = policy_fn(state)
        reward = env.step(action)
        total_reward += reward

    return total_reward / steps

def test(steps = 10000):

    # -- Test Slots:
    slots = [
        {"outcomes": [500, 0], "probabilities": [0.8, 0.2]},
        {"outcomes": [0, 100], "probabilities": [0.2, 0.8]},
    ]

    # -- Training Slots:
    # SLOTS = [
    #     {"outcomes": [10, 0], "probabilities": [0.8, 0.2]},
    #     {"outcomes": [0, 100], "probabilities": [0.2, 0.8]},
    # ]

    # Create test env
    test_env = BanditsEnv(slots)

    avg_learned = evaluate(test_env, greedy_policy, steps)
    avg_random = evaluate(test_env, random_policy, steps)

    print(f"Average reward (learned): {avg_learned:.3f}")
    print(f"Average reward (random):  {avg_random:.3f}")


test(steps=10000)


# ------ Graphs
q_history = np.array(q_history)

plt.figure(figsize=(8, 5))
for i in range(q_history.shape[1]):
    plt.plot(q_history[:, i], label=f"Q(slot {i})")

plt.xlabel("Training checkpoints")
plt.ylabel("Q-value")
plt.title(f"DQN learning on bandit")
plt.legend()
plt.tight_layout()
plt.show()