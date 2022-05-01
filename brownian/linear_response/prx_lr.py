import torch
import numpy as np
import math
from sde_op import SDEOP
import torch.nn as nn
import torchsde
from utils import compute_dissipation, compute_heat, compute_work, compute_energy


class Protocol:
    def __init__(self, d, t_h, t_c, var_h, var_l, num_intervals,
                 dt=0.001, device=torch.device("cpu"),
                 num_particles=1000, sde_type="ito", folder_name="./"):

        assert t_h > t_c and var_h > var_l
        self.device = device
        self.num_particles = num_particles
        self.d = d
        k_h = t_h/var_l
        k_l = t_c/var_h
        self.num_intervals = num_intervals
        self.gamma_q = self._get_gamma_q()
        self.temp_protocol = torch.tensor(self._get_temp_protocol(t_h=t_h,
                                                                  t_c=t_c),
                                          device=device).float()
        self.k_protocol = torch.tensor(self._get_k_protocol(k_h=k_h,
                                                            k_l=k_l),
                                       device=device).float()
        self.dt = dt
        self.sde = SDEOP(num_particles=num_particles,
                         sde_type=sde_type).to(device)
        self.folder_name = folder_name

    def _initialize_position(self):
        """Initializes positions based on normal distribution with mean 0
        and variance self.var_init[0]
        Returns:
            particle_position: A tensor (self.num_particles, 1)
            representing the particle positions.
        """

        return (torch.randn((self.num_particles, 1), device=self.device)
                * math.sqrt((self.temp_protocol[0]/self.k_protocol[0]).item()))

    def _get_gamma_q(self):
        """Compute gamma_q as defined in Eq. 80
        """
        x = np.arange(self.num_intervals)/self.num_intervals
        y = (((np.sqrt(1 + self.d) * np.sin(2*np.pi * x))
              / (2*np.sqrt((np.sin(2*np.pi*x))**2 + self.d)))
             + 0.5)
        return y

    def _get_temp_protocol(self, t_h, t_c):
        """Gets the temperature protocol Eq.9
        """
        return ((t_c*t_h)/(t_h - (t_h-t_c)*self.gamma_q))

    def _get_k_protocol(self, k_h, k_l):
        """Gets the protocol of harmonic potential Eq. 67 and Eq.74
        """
        gamma_q_diff = (np.cumsum((self.gamma_q - np.mean(self.gamma_q)))
                        / self.num_intervals)
        y = ((0.5 * self.gamma_q - (0.5)*gamma_q_diff))
        scale = (k_h - k_l) / (y.max() - y.min())
        shift = k_l - scale * y.min()
        y = y * scale + shift
        return y

    def _save_test_data(self, loss, dissipation, work, heat, energy, mean, var):
        filename = self.folder_name + "test_loss.npy"
        np.save(filename, np.array(loss))

        filename = self.folder_name + "test_dissipation.npy"
        np.save(filename, np.array(dissipation))

        filename = self.folder_name + "test_work.npy"
        np.save(filename, np.array(work))

        filename = self.folder_name + "test_heat.npy"
        np.save(filename, np.array(heat))

        filename = self.folder_name + "test_energy.npy"
        np.save(filename, np.array(energy))

        filename = self.folder_name + "test_mean.npy"
        np.save(filename, np.array(mean))

        filename = self.folder_name + "test_var.npy"
        np.save(filename, np.array(var))

    def test(self, time_per_cycle, interval_length_, num_iterations=1000):
        """Carries out protocol described in Seifert  
        Args:
            time_per_cycle: The time length of each cycle (tau)
            interval_length_: Number of time steps between each step of optimization. This is included to make 
                              sure calculations were done correctly
            num_iterations: An int representing the number of iterations of the current protocol to run 

        Saves:
            test_loss.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                           the loss at each interval of optimization
            test_var.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                          the variance of particles at each interval of optimization
            test_mean.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                           the mean of particles at each interval of optimization
            test_heat.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing
                           the total heat produced at each interval
            test_work.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing
                            the total work produced at each interval
            test_energy.npy: A numpy array of shape (num_epochs) representing the total 
                             energy produced per cycle
            test_dissipation.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing
                                  the total entropy prduction at each interval
            str(epoch) + _all_y.pt: A numpy array of shape (num_steps, num_intervals_per_step) representing 
                                    the current positions

        """
        time_per_interval = time_per_cycle / self.num_intervals
        interval_length = int(time_per_interval / self.dt)
        assert interval_length == interval_length_

        init_pos = self._initialize_position()
        all_loss = []
        all_dissipation = []
        all_work = []
        all_heat = []
        all_energy = []
        all_mean = []
        all_var = []

        # Temporarily set the initial protocol to be the last [T, k]
        # Only matters for work computation for the first interval of the first epoch
        temp, k = self.temp_protocol[-1], self.k_protocol[-1]
        ts = torch.arange(0, time_per_interval + self.dt,
                          self.dt, device=self.device)
        for epoch in range(num_iterations):
            all_loss.append([])
            all_dissipation.append([])
            all_work.append([])
            all_heat.append([])
            all_mean.append([])
            all_var.append([])
            all_y = []
            init_cycle_pos = init_pos.clone()
            for step in range(self.num_intervals):
                old_temp, old_k = temp, k
                temp, k = self.temp_protocol[step], self.k_protocol[step]

                self.sde.set_T_k(T=temp, k=k)
                ts = ts.detach()  # This step is probably superfluous
                ys = torchsde.sdeint(self.sde, init_pos, ts,
                                     dt=self.dt, method='euler')

                init_pos = ys[-1].detach()
                mean_count = ys[-2].mean()
                var_count = ys[-2].var()

                all_mean[-1].append(mean_count.item())
                all_var[-1].append(var_count.item())
                all_y.append(init_pos.tolist())

                l = (nn.MSELoss()(mean_count,
                                  torch.zeros_like(mean_count))
                     + nn.MSELoss()(var_count,
                                    temp/k))
                all_loss[-1].append(l.item())

                dissipation = compute_dissipation(ys, temp, k)
                work = compute_work(ys[0], old_k, k)
                heat = compute_heat(ys, k)
                all_dissipation[-1].append(dissipation)
                all_work[-1].append(work)
                all_heat[-1].append(heat)
                del ys, mean_count, var_count, l
            torch.save(all_y, str(epoch) + "_all_y.pt")
            # Computing the change in energy for k at the very end of the cycle
            energy_change = compute_energy(k=self.k_protocol[0],
                                           pos_i=init_cycle_pos, pos_f=init_pos)
            all_energy.append(energy_change)
            self._save_test_data(all_loss, all_dissipation, all_work,
                                 all_heat, all_energy, all_mean, all_var)
