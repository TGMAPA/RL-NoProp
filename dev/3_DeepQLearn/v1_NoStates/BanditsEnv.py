import random


class BanditsEnv:
    def __init__(self, slots):
        """
        slots: list of dicts with keys:
            - 'outcomes'
            - 'probabilities'
        """
        self.slots = slots

        self.num_actions = len(slots)

        self.state = [1.0]

        
    # Execute an action (pull a slot)
    def step(self, action):
        slot = self.slots[action]
        reward = random.choices(
            slot["outcomes"],
            slot["probabilities"]
        )[0]

        # No transition

        return reward
