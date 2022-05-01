import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class LambdaHarmonic(nn.Module):
    def __init__(self, tau):
        """
        Args:
            tau: The time length of each cycle
        """
        super().__init__()
        self.register_buffer("pi", torch.tensor(np.pi))
        self.register_buffer("tau", torch.tensor(tau))

    def forward(self, t):
        """Computes the protocol for a harmonic protocol
        Args:
            t: A torch tensor of shape (num_steps, 1) representing the time steps for which to determine a protocol
        Returns:
            protocol: A tensor of shape (num_steps, 2) representing the (T, V) along each step
        """
        protocol = torch.cat([1 + (torch.sin(self.pi * t/self.tau) ** 2),
                              1 + (torch.sin(self.pi * t/self.tau + self.pi/4) ** 2)], dim=-1)
        return protocol


class LambdaLinear(nn.Module):
    def __init__(self, tau):
        """
        Args:
            tau: The time length of each cycle
        """
        super().__init__()
        self.register_buffer("pi", torch.tensor(np.pi))
        self.register_buffer("tau", torch.tensor(tau))

    def forward(self, t):
        """Computes the protocol for a linear protocol
        Args:
            t: A tensor of shape (num_steps, 1) representing the time steps for which to determine a protocol
        Returns:
            protocol: A tensor of shape (num_steps, 2) representing the (T, V) along each step
        """
        t = t/self.tau
        protocol = (((t < 0.25) * torch.cat([1 + t * 2,
                                             1.5 + t * 2], dim=-1))
                    + ((0.25 <= t) * (t < 0.5) * torch.cat([1.5 + (t - 0.25) * 2,
                                                            2 - (t - 0.25) * 2], dim=-1))
                    + ((0.5 <= t) * (t < 0.75) * torch.cat([2 - (t - 0.5) * 2,
                                                            1.5 - (t - 0.5) * 2], dim=-1))
                    + ((0.75 <= t) * (t <= 1.00) * torch.cat([1.5 - (t - 0.75) * 2,
                                                              1. + (t - 0.75) * 2], dim=-1)))
        return protocol
