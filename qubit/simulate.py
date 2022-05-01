from locale import T_FMT
from matplotlib.pyplot import step
import torch
from torch.autograd import grad


class Protocol:
    def __init__(self, lambda_, dt, device=torch.device("cpu")):
        self.device = device
        self.lambda_ = lambda_
        self.T = self.V = self.T_dot = self.V_dot = self.k_plus = self.k_minus = self.theta = self.L = self.L_b = None
        self.dt = dt

    def get_k(self, V, T, is_plus=True, gamma=5):
        """Gets k as defined in Equation
        Args:
            V: A tensor of shape (num_time_steps, 1) corresponding to the level-splitting
            T: A tensor of shape (num_time_steps, 1) corresponding to the temperature
            is_plus: A bool to determine whether to compute k^+ or k^-
            gamma: A float representing the average jump frequency
        """
        if is_plus:
            return (gamma * V * (1 + torch.exp(V/T)))/(torch.exp(V/T) - 1)
        else:
            return (gamma * V * (1 - torch.exp(V/T)))/(torch.exp(V/T) - 1)

    def get_density(self, state, V, epsilon=0.6):
        """Computes the density matrix for a given state and V (level-splitting) using Eq. 
        Args:
            state:
            V: A tensor of shape (num_time_steps, 1) corresponding to the level-splitting
            epsilon: A float representing the coherence parameter
        """
        # Eq.
        denominator = torch.sqrt(2 * (V ** 2)
                                 + (2 * V * torch.sqrt(V ** 2 - epsilon ** 2)))
        sinvt = epsilon / denominator
        cosvt = (V + torch.sqrt(V ** 2 - epsilon ** 2)) / denominator
        evp = torch.cat((sinvt, -cosvt))
        evm = torch.cat((cosvt, sinvt))

        i = torch.complex(torch.tensor(0.), torch.tensor(1.)).to(self.device)
        pi_x = torch.outer(evp, evm) + torch.outer(evm, evp)
        # removing i's to avoid pytorch error (with torch.stack)during backprop
        pi_y = torch.outer(evm, evp) - torch.outer(evp, evm)
        pi_z = torch.outer(evp, evp) - torch.outer(evm, evm)
        all_pi = torch.stack([pi_x, pi_y, pi_z]).unsqueeze(0)
        c = 0.5 * torch.eye(2).to(self.device)
        state = state.unsqueeze(-1).unsqueeze(-1)
        density = (state * all_pi).sum(1) + c
        return density

    def get_theta(self, V, epsilon=0.6):
        """Computes theta in Eq. 
        Args:
            V: A tensor of shape (num_time_steps, 1) corresponding to the level-splitting
            epsilon: A float representing the coherence parameter
        """
        return epsilon/(V * torch.sqrt(V**2 - epsilon**2))

    def update_protocol(self, protocol_step, ts):
        """Computes the current protocol given the protocol_step and a time interval
        Args:
            protocol_step: An int representing the step number of the protocol
            ts: A tensor of shape (interval_length, 1) representing the time steps
        Returns:
            gradient_deviation: A tensor of shape (1) representing the total deviation from a 
                                linear gradient. Used for a potential regularization
        Modifies:
            self.T: A tensor of shape (interval_length, 1) representing the current temperature of the interval
            self.V: A tensor of shape (interval_length, 1) representing the current level-splitting of the interval
            self.theta: A tensor of shape (interval_length, 1) representing the current theta of the interval
            self.T_dot: A tensor of shape (interval_length, 1) representing the time derivative of the current temperature of the interval
            self.V_dot: A tensor of shape (interval_length, 1) representing the time derivative of the current level-splitting of the interval
            self.L: A tensor of shape (interval_length, 3, 3) representing the propagator for the interval
            self.L_b: A tensor of shape (interval_length, 3) representing the constant of the propagator for the interval
            self.k_plus: A tensor of shape (interval_length, 1) representing the current k^plus of the interval
            self.k_minus: A tensor of shape (interval_length, 1) representing the current k^minus of the interval 
        """
        ts.requires_grad = True
        lmb = self.lambda_(ts)
        T = lmb[:, 0].unsqueeze(-1)
        V = lmb[:, 1].unsqueeze(-1)
        V_dot = grad(V, ts, grad_outputs=torch.ones_like(V),
                     create_graph=True, retain_graph=True)[0]
        T_dot = grad(T, ts, grad_outputs=torch.ones_like(T),
                     create_graph=True, retain_graph=True)[0]
        self.T_dot = T_dot
        self.V_dot = V_dot
        T_linear, V_linear = self.lambda_.get_linear_gradient(protocol_step)
        gradient_deviation = (torch.abs((self.V_dot - V_linear)/V_linear)
                              + torch.abs((self.T_dot - T_linear)/T_linear))
        gradient_deviation = gradient_deviation.mean()
        self.T = T
        self.V = V

        self.k_minus = self.get_k(self.V, self.T, is_plus=False)
        self.k_plus = self.get_k(self.V, self.T, is_plus=True)
        self.theta = self.get_theta(self.V)
        self.L = torch.stack([torch.cat([-self.k_plus, -self.V, self.theta*self.V_dot],
                                        dim=-1),
                              torch.cat([self.V, -self.k_plus, torch.zeros_like(self.V, device=self.device)],
                              dim=-1),
                              torch.cat([self.theta * self.V_dot, torch.zeros_like(self.V_dot, device=self.device), -2 * self.k_plus],
                              dim=-1)],
                             dim=-1)

        self.L_b = torch.cat([torch.zeros_like(self.k_minus,
                                               device=self.device),
                              torch.zeros_like(self.k_minus,
                                               device=self.device),
                              self.k_minus], dim=-1)

        return gradient_deviation

    def take_step(self, step_num, curr_state, get_heat_work=False):
        """Takes an integration step as described in S48. Computes the heat and work and described in 
           Equation 
        Args:
            step_num: An int representing the current step within the interval (not the protocol step!)
            curr_state: A tensor of shape (3) representing the current state of the system
            get_heat_work: A bool determining whether to compute the heat and work
        Returns:
             next_state: A tensor of shape (3) representing the next state of the system
             density: A tensor of shape (1, 2, 2) representing the current density of the system
             dq: A tensor of shape (1) representing the heat produced for the time step
             dw: A tensor of shape (1) representing the work produced for the time step
             disipation: A tensor of shape (1) representing the entropy produced for the time step
        """
        L_step = self.L[step_num]
        L_b_step = self.L_b[step_num]
        V_step = self.V[step_num]
        T_step = self.T[step_num]
        theta_step = self.theta[step_num]
        V_dot_step = self.V_dot[step_num]
        T_dot_step = self.T_dot[step_num]

        dr = L_step @ curr_state + L_b_step
        next_state = (curr_state + self.dt * dr)
        density = self.get_density(next_state, V_step)

        if get_heat_work:
            with torch.no_grad():
                r = torch.linalg.norm(curr_state)
                r_x, _, r_z = curr_state
                dw = -(-V_step*theta_step*r_x + r_z) * V_dot_step * self.dt
                dq = 0.5 * ((4 * r * torch.atanh(2 * r) +
                            torch.log(0.25 - r ** 2)) * T_dot_step * self.dt)
                dissipation = dq/T_step
            return next_state, dw, dq, dissipation, density
        else:
            return next_state, density

    def get_dqW(self, t):
        """Computes ADIABATIC WORK
        """
        t.requires_grad = True
        T, V = self.lambda_(t)
        V_dot = grad(V, t)[0]
        dqw = 0.5 * (torch.tanh(V/(2*T)) * V_dot)
        return dqw


class Simulate:
    def __init__(self, lambda_, dt, device=torch.device("cpu")):
        self.device = device
        self.protocol = Protocol(lambda_, dt, device=device)

    def simulate(self, init_state, init_density, ts, interval_length, protocol_step, track_gradients=True):
        """Simulates protocol
        Args:
            init_state: A tensor of shape (2) representing the current state of the system
            ts: A tensor of shape (interval_length, 1) representing the time steps
            interval_length: The number of simulation steps
            protocol_step: The step number of the protocol
            track_gradients: A bool representing whether to track gradients
        Returns:
            all_states: A list of length_interval containing tensors of shape (3) 
                        representing the current state along the interval
            all_densities: A list of length interval_length containing tensors of shape (2, 2) 
                           representing the densities along 
            gradient_deviation: A tensor of shape () represesnting the gradient_deviation term
        """
        assert ts.shape[0] == interval_length
        all_states = [init_state]
        all_densities = [init_density]
        gradient_deviation = self.protocol.update_protocol(protocol_step=protocol_step,
                                                           ts=ts)
        curr_state = init_state
        for step_num in range(interval_length):
            curr_state, curr_density = self.protocol.take_step(step_num,
                                                               curr_state,
                                                               get_heat_work=False)
            if not track_gradients:
                curr_state = curr_state.detach()
                curr_density = curr_density.detach()
            all_states.append(curr_state)
            all_densities.append(curr_density[0])

        return all_states, all_densities, gradient_deviation

    def simulate_test(self, init_state, init_density, ts, interval_length, protocol_step):
        """Simulates protocol and also computes heat, work and dissipation. Doesn't track gradients
        Args:
            init_state: A tensor of shape (2) representing the current state of the system
            ts: A tensor of shape (interval_length, 1) representing the time steps
            interval_length: The number of simulation steps
            protocol_step: An int representing the step number of the protocol
        Returns:
            all_states: A tensor of shape (interval_length + 1, 3) representing the states of the system along the interval
            all_work: A list of length (num_steps) containing tensors of shape (1) representing the work of the system along the interval
            all_heat: A list of length (num_steps) containing tensors of shape (1) representing the heat of the system along the interval
            all_dissipation: A list of length (num_steps) containing tensors of shape (1) representing the dissipation of the system along the interval
            all_densities: A tensor of shape (interval_length + 1, 2, 2) representing the densities of the system along the interval
        """
        assert ts.shape[0] == interval_length
        all_work = []
        all_heat = []
        all_dissipation = []
        all_states = [init_state]
        all_densities = [init_density]

        gradient_deviation = self.protocol.update_protocol(protocol_step=protocol_step,
                                                           ts=ts)
        curr_state = init_state
        for step_num in range(interval_length):
            all_states.append(curr_state)
            curr_state, dw, dq, dissipation, curr_density = self.protocol.take_step(step_num,
                                                                                    curr_state,
                                                                                    get_heat_work=True)

            curr_state = curr_state.detach()
            curr_density = curr_density.detach()
            all_heat.append(dq)
            all_work.append(dw)
            all_dissipation.append(dissipation)
            all_densities.append(curr_density.detach())
        all_states = torch.stack(all_states)
        all_densities = torch.cat(all_densities)
        
        return all_states, all_work, all_heat, all_dissipation, all_densities
