import random
import torch


class BanditsEnv:
    def __init__(self, slots):
        """
        slots: list of dicts with keys:
            - 'outcomes'
            - 'probabilities'
        """
        self.slots = slots

        self.num_actions = len(slots)

        self.states = {
            0: torch.tensor([1,0,0,0, 0,1,0,0], dtype=torch.float32),
            1: torch.tensor([0,1,0,0, 1,0,0,0], dtype=torch.float32)
        }

    # Get random state
    def getState(self):
        self.state_id = random.choice([0, 1])
        return self.states[self.state_id]

    # Execute an action (pull a slot)
    def step(self, action):
        slot = self.slots[action]
        reward = random.choices(
            slot["outcomes"],
            slot["probabilities"]
        )[0]

        # No transition

        return reward
