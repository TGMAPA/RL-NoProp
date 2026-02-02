import torch

class Diffusion:
    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha

    def diffuse(self, q_target):
        z = [q_target]
        for t in range(self.T):
            eps = torch.randn_like(q_target)
            z_t = torch.sqrt(self.alpha[t]) * z[-1] + torch.sqrt(1 - self.alpha[t]) * eps
            z.append(z_t)
        return z