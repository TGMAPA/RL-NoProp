
from DQN import DQN
from BanditsEnv import BanditsEnv
from ReplayMemory import ReplayMemory
import torch


# -- Config
BUFFER_CAPACITY = 1000

SLOTS = [
    {"outcomes": [0, 10], "probabilities": [0.6, 0.4]},
    {"outcomes": [0, 100], "probabilities": [0.97, 0.03]},
]

TARGET_UPDATE = 100

BATCH_SIZE = 32

LR = 1e-3



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

