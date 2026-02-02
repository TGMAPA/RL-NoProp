import torch
import torch.nn as nn

class DenoisingMLP(nn.Module):

    def __init__(self, n_actions):
        """
        Parameters:
            in_channels: int
            n_actions: int
                The number of actions we want to predict.
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_actions + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
        

    def forward(self, state, z_t):
        """
        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = torch.cat([state, z_t], dim=1) 
        
        return self.fc(x)