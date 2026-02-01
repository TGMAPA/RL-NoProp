import random
from collections import deque


class ReplayMemory:

    # Initialize que with any desired capacity
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Save a transition
    def push(self, transition):
        self.buffer.append(transition)

    # Get a random sample form buffer
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        return sample

    # Get buffer's lenght
    def __len__(self):
        return len(self.buffer)