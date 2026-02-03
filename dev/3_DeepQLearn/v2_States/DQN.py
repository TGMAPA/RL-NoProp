import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, in_channels, n_actions):
        """
        Parameters:
            in_channels: int
            n_actions: int
                The number of actions we want to predict.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(in_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_actions)

    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))               # FC1 + ReLU
        x = F.relu(self.fc2(x))               # FC2 + ReLU
        x = self.fc3(x)                       # FC3 (output)
        
        return x