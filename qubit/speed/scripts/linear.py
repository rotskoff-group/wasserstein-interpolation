import argparse
import os
import torch
from qubit_lr import BaseProtocol

folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=1.)
config = parser.parse_args()
tau = config.tau
os.mkdir("./" + str(tau))

b = BaseProtocol(tau = tau, folder_name="./" + str(tau) + "/", use_harmonic=False)
num_steps = round(tau/b.dt)
phi_dot = torch.ones(((num_steps, 1)))
phi  = torch.arange(0, tau, b.dt).unsqueeze(-1)
if tau < 0.5:
    burn_in_steps=3
else:
    burn_in_steps=1
b.test(phi = phi, phi_dot = phi_dot, burn_in_steps=burn_in_steps)