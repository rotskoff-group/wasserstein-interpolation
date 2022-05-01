import argparse
import os
import torch
import numpy as np
from qubit_lr import BaseProtocol
from scipy.interpolate import interp1d

harmonic_density_file_name = "/home/shriramc/Documents/outputs/040922QubitBaseProtocols/harmonic/50.0/test_densities.npy"
folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=1.)
config = parser.parse_args()
tau = config.tau
os.mkdir("./" + str(tau))

b = BaseProtocol(tau = tau, folder_name="./" + str(tau) + "/")
t = torch.arange(0, tau, b.dt)
density = torch.tensor(np.load(harmonic_density_file_name)).float().diagonal(dim1=-2, dim2=-1)
dl = torch.linalg.norm(torch.diff(density, dim=0), dim=-1)
L = dl.sum()
frac_dl = (dl/(L))
phi_dot = (1/(frac_dl * 50) * b.dt)
effective_time = torch.cat((torch.tensor([0]), torch.cumsum(frac_dl, dim=0)))[:-1]
phi_dot_interpolation = interp1d(effective_time.numpy(), phi_dot, bounds_error=False, fill_value="extrapolate")
phi_dot = phi_dot_interpolation(t/tau)
phi_dot = torch.tensor(phi_dot)
phi = torch.cumsum(phi_dot * torch.diff(t/tau, prepend=torch.tensor([0.])), dim=-1)
phi = phi.unsqueeze(-1) * tau
phi_dot = phi_dot.unsqueeze(-1)

if tau < 0.5:
    burn_in_steps=3
else:
    burn_in_steps=1
b.test(phi = phi, phi_dot = phi_dot, burn_in_steps=burn_in_steps)