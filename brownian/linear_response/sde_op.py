import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SDEOP(nn.Module):
    def __init__(self, num_particles=1000, sde_type="ito"):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = sde_type
        self.num_particles=num_particles
        self.dimensions = 1
        self.T = None
        self.k = None

    def set_T_k(self, T, k):
        self.T = T
        self.k = k


    def f(self, t, y):
        """Drift Calculation
        Arguments:
            t: A tensor of shape () representing the time of the simulation
            y: A tensor of shape (self.num_particles, self,dimensions)
            representing the particle positions
        Returns:
            force: A tensor of shape (self.num_particles, self.dimensions)
            representing the force on each particle
        """
        force = -self.k * y
        return force

    def g(self, t, y):
        """Diffusion Calculation
        Arguments:
            t: A tensor of shape () representing the time of the simulation
            y: A tensor of shape (self.num_particles, self.dimensions)
            representing the particle positions
        Returns:
            diffusion: A tensor of shape (self.num_particles, self.dimensions)
            representing the diffusion constant of each particle
        """
        diffusion = torch.sqrt(2 * self.T)
        return diffusion.repeat(self.num_particles * self.dimensions).view(self.num_particles, self.dimensions)
