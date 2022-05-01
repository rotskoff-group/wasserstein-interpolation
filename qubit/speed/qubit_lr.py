import torch
from torch.autograd import grad
import numpy as np
from simulate import Simulate
from networks import LambdaHarmonic, LambdaLinear


class BaseProtocol:
    def __init__(self, tau, dt=0.001, use_harmonic=True,
                 device=torch.device("cpu"), folder_name="./"):
        """
        Args:
            tau: A float representing the time of the protocol
            use_harmonic: A boolean for whether to use a harmonic protocol or a linear protocol
            dt: The time step of integration
        """
        self.device = device
        self.dt = dt
        self.tau = tau
        if use_harmonic:
            self.lambda_ = LambdaHarmonic(tau)
        else:
            self.lambda_ = LambdaLinear(tau)
        self.simulate = Simulate(lambda_=self.lambda_, dt=self.dt,
                                 device=device)
        self.folder_name = folder_name

    def get_protocol(self):
        """Returns the current protocol
        Returns:
            T: A tensor of shape (num_steps, 1) representing the current temperature of the cycle
            V: A tensor of shape (num_steps, 1) representing the current level-splitting of the cycle
            T_dot: A tensor of shape (num_steps, 1) representing the time derivative of the current temperature of the cycle        
            V_dot: A tensor of shape (num_steps, 1) representing the time derivative of the current level-splitting of the cycle
        """
        ts = torch.arange(0, self.tau, self.dt).unsqueeze(-1)
        ts.requires_grad = True
        lmb = self.lambda_(ts)
        T = lmb[:, 0].unsqueeze(-1)
        V = lmb[:, 1].unsqueeze(-1)
        V_dot = grad(V, ts, grad_outputs=torch.ones_like(V),
                     create_graph=False, retain_graph=True)[0]
        T_dot = grad(T, ts, grad_outputs=torch.ones_like(T),
                     create_graph=False, retain_graph=False)[0]

        return T.detach(), V.detach(), T_dot.detach(), V_dot.detach()

    def test(self, phi, phi_dot, burn_in_steps=1):
        """Simulates the protocol for a given speed function
        Args:
            phi: A tensor of shape (num_steps, 1) representing the time of the protocol
            phi_dot: A tensor of shape (num_steps, 1) representing the 
                     "speed" of the protocol
            burn_in_steps: The number of steps for which to equilibrate the protocol
        """
        num_steps = round(self.tau/self.dt)
        assert phi_dot.shape[0] == num_steps

        # Initial state doesnt matter
        init_state = torch.rand((3))
        init_state /= (init_state.sum() * 2)
        init_density = torch.randn((1, 2, 2))  # Not necessary to compute this
        for period in range(burn_in_steps + 1):
            is_burn_in_stage = (period < burn_in_steps)
            period_states, period_work, period_heat, period_dissipation, period_densities, period_q_work = self.simulate.simulate_test(init_state=init_state,
                                                                                                                                       init_density=init_density,
                                                                                                                                       ts=phi,
                                                                                                                                       phi_dot=phi_dot,
                                                                                                                                       num_steps=num_steps)
            init_state = period_states[-1].detach()
            init_density = period_densities[-1].detach().unsqueeze(0)
            if (is_burn_in_stage):
                continue

        period_states = period_states.cpu().numpy()
        period_densities = period_densities.cpu().numpy()
        period_work = period_work.cpu().numpy()
        period_heat = period_heat.cpu().numpy()
        period_dissipation = period_dissipation.cpu().numpy()
        period_q_work = period_q_work.cpu().numpy()

        filename = self.folder_name + "test_states.npy"
        np.save(filename, period_states)
        filename = self.folder_name + "test_densities.npy"
        np.save(filename, period_densities)
        filename = self.folder_name + "test_work.npy"
        np.save(filename, period_work)
        filename = self.folder_name + "test_heat.npy"
        np.save(filename, period_heat)
        filename = self.folder_name + "test_dissipation.npy"
        np.save(filename, period_dissipation)
        filename = self.folder_name + "test_q_work.npy"
        np.save(filename, period_q_work)
