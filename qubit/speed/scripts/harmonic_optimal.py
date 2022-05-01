import argparse
import os
import torch
from qubit_lr import BaseProtocol
from scipy.interpolate import interp1d

folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=1.)
config = parser.parse_args()
tau = config.tau
os.mkdir("./" + str(tau))


def get_dl(T, V, T_dot, V_dot, dt, tau):
    """
    Args:
        T: A tensor of shape (num_steps) representing the current temperature of the cycle
        V: A tensor of shape (num_steps) representing the current level-splitting of the cycle
        T_dot: A tensor of shape (num_steps) representing the time derivative of the current temperature of the cycle        
        V_dot: A tensor of shape (num_steps) representing the time derivative of the current level-splitting of the cycle
        dt: A float representing the time step
        tau: A float representing the time length of the protocol
    Returns:
        dl: A tensor of shape (num_steps) representing the "distance" at each interval
        L: A tensor of shape () represnting the total length of the protocol
    """
    
    lmb_dot = torch.stack((T_dot, V_dot))
    lmb_dot = lmb_dot.T
    
    
    gamma = 5
    epsilon = 0.6
    R_factor = torch.stack([torch.stack([-V/(T**3), 1/(T**2)]), torch.stack([1/(T**2), -1/(V*T)])])
    R_lambda = (1/(8 * gamma)) * (torch.tanh(V/(2*T))/(torch.cosh(V/(2*T)) ** 2))
    coth = 1/(torch.tanh(V/(2*T)))

    C = -((epsilon**2) / ((2 * V**2) * (V**2 - epsilon**2))) * (gamma/(((gamma*coth)**2) + 1))
    C = torch.stack([torch.stack([torch.zeros_like(C), torch.zeros_like(C)]), torch.stack([torch.zeros_like(C), C])])
    R = R_factor * R_lambda + C
    R_t = torch.transpose(R, dim0=0, dim1=1)
    g = -0.5*(R + R_t)
    g = g.permute((-1, 0, 1))

    dl = torch.sqrt(torch.bmm(lmb_dot.unsqueeze(-2), torch.bmm(g, lmb_dot.unsqueeze(-1)))) 
    dl = dl.flatten() * dt
    
    L = dl.sum()
    
    dA = (torch.bmm(lmb_dot.unsqueeze(-2), torch.bmm(g, lmb_dot.unsqueeze(-1))))
    dA = dA.flatten() * dt
    A = dA.sum()
    assert A * tau - L**2 >= 0
    
    return dl, L

b = BaseProtocol(tau = tau, folder_name="./" + str(tau) + "/")
t = torch.arange(0, tau, b.dt)
T, V, T_dot, V_dot = b.get_protocol()
T = T.flatten()
V = V.flatten()
T_dot = T_dot.flatten()
V_dot = V_dot.flatten()
dl, L = get_dl(T, V, T_dot, V_dot, dt=b.dt, tau=tau)
frac_dl = (dl/(L))
phi_dot = (1/(frac_dl * tau) * b.dt)
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