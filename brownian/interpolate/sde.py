import torch
import torch.nn as nn


class SDE(nn.Module):
    def __init__(self, lambda_, num_particles=1000, sde_type="ito"):

        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = sde_type
        self.num_particles = num_particles
        self.dimensions = 1
        self.register_buffer('t_frac', torch.tensor([0.]))
        self.protocol_step = 0
        self.lambda_ = lambda_

    def set_t_frac(self, t_frac):
        """Updates the fraction (i.e. t_protocol/tau_protocol) of the protocol along a step
        """
        self.t_frac = t_frac

    def set_protocol_step(self, protocol_step):
        """Sets the current protocol step
        """
        self.protocol_step = protocol_step

    def get_T_k(self):
        """Gets temperature and k constant, representing strength of harmonic_potential
        Returns:
            temp: A float representing the current temperature
            k: A float representing the current spring constant
        """
        
        temp, k = self.lambda_(self.protocol_step, self.t_frac).tolist()
        return temp, k

    def f(self, t, y):
        """Drift Calculation
        Arguments:
            t: A tensor of shape () representing the time of the simulation
            y: A tensor of shape (self.num_particles, self,dimensions)
            representing the particle positions
        Returns:
            force: A tensor of shape (self.num_particles, self.dimensions)
            representing the force on each particle
        Raises:
            ValueError if a negative k constant is predicted
        """
     
        k = self.lambda_(self.protocol_step, self.t_frac)[1]
        if (k < 0):
            raise ValueError("Negative k constant")
        force = -k * y
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
        Raises:
            ValueError if a negative temperature is predicted
        """
        temp = self.lambda_(self.protocol_step, self.t_frac)[0]
        if (temp < 0):
            raise ValueError("Negative Temperature")
        diffusion = torch.sqrt(2 * temp)
        diffusion = diffusion.repeat((self.num_particles, self.dimensions))
        return diffusion
