from DQN import DQN
from BanditsEnv import BanditsEnv
from ReplayMemory import ReplayMemory
from TrainingLoop import train
import torch, random


# -- Config
BUFFER_CAPACITY = 1000

SLOTS = [
    {"outcomes": [0, 10], "probabilities": [0.6, 0.4]},
    {"outcomes": [0, 100], "probabilities": [0.97, 0.03]},
]

TARGET_UPDATE = 100

BATCH_SIZE = 32

LR = 1e-3

EPISODES = 5000



# -- Initialize environment
ENV = BanditsEnv(SLOTS)


# -- Initialize online and target network
# Trainable Net
onlineNet = DQN(1, ENV.num_actions)

# Target Net
targetNet = DQN(1, ENV.num_actions)

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
train(
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


# # Test
# with torch.no_grad():
#     q_final = onlineNet(torch.tensor([[1.0]]))

# print("Learned q-values: ", q_final)



def greedy_policy(state):
    with torch.no_grad():
        q_values = onlineNet(state)
        return q_values.argmax(dim=1).item()

def random_policy(_):
    return random.randint(0, ENV.num_actions - 1)

def evaluate(policy_fn, steps=10000):
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
