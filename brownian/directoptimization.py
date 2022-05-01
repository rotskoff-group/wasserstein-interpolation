import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsde
from scipy.stats import skew
import os
from networks import Model
from sde import SDE
from utils import compute_dissipation, compute_heat, compute_work, compute_energy


class DirectOptimization:
    def __init__(self, t_h, t_c, var_h, var_l, dt=0.001, device=torch.device("cpu"),
                 num_particles=1000, sde_type="ito", folder_name="./"):
        assert t_h > t_c and var_h > var_l
        self.device = device
        self.num_particles = num_particles
        self.var = [var_l, var_h, var_h, var_l]
        self.protocol = [[t_h, t_h/self.var[0]],
                         [t_h, t_h/self.var[1]],
                         [t_c, t_c/self.var[2]],
                         [t_c, t_c/self.var[3]]]
        self.is_isothermal = [True, False, True, False]

        self.model = Model(protocol=self.protocol, is_isothermal=self.is_isothermal,
                           device=self.device)
        self.sde = SDE(lambda_=self.model.lambda_,
                       num_particles=num_particles,
                       sde_type=sde_type).to(device)

        self.dt = dt
        self.folder_name = folder_name

    def _initialize_position(self):
        """Initializes positions based on normal distribution with mean 0
        and variance self.var_init[0]
        Returns:
            particle_position: A tensor (self.num_particles, 1)
            representing the particle positions.
        """

        return (torch.randn((self.num_particles, 1), device=self.device) * math.sqrt(self.var[0]))

    def _get_mean_var(self, t_frac, var_init, var_final, mean_init=0,
                      mean_final=0):
        """Gets the target mean and variance for a given t_frac
        Arguments:
            t_frac: A Float representing the t fraction
            var_init: A Float representing the inital variance of the step
            var_init: A Float representing the final variance of the step
            mean_init: A Float representing the inital mean of the step
            mean_final: A Float representing the final mean of the step
        Returns:
            target_mean: A torch tensor of shape () representing the target mean
            target_var: A torch tensor of shape () representing the target var
        """
        target_mean = mean_final * t_frac + mean_init * (1 - t_frac)
        target_var = (math.sqrt(var_final) * (t_frac)
                      + math.sqrt(var_init) * (1 - t_frac)) ** 2
        return (torch.tensor(target_mean, device=self.device),
                torch.tensor(target_var, device=self.device))

    def plot_statistics(self, y, folder_name="./", tag=""):
        """Plots distribution of particle positions
        Arguments:
            y: Torch tensor of shape (self.num_particles, 1) representing
            particle positions
            folder_name: Folder to save image in
            tag: A string for the label of the image file
        """
        y = y.cpu().numpy()
        plt.hist(y, density=True)
        mean = round(np.mean(y), 4)
        var = round(np.var(y), 4)
        sk = round(skew(y), 4)
        title = "MEAN: " + str(mean) + " Var: " + str(var) + " Skew:" + str(sk)
        plt.title(title)
        plt.savefig(folder_name + tag + ".png")
        plt.close()

    def _save_training_data(self, iteration, loss, var, mean):
        filename = self.folder_name + "loss.npy"
        np.save(filename, np.array(loss))

        filename = self.folder_name + "var.npy"
        np.save(filename, np.array(var))

        filename = self.folder_name + "mean.npy"
        np.save(filename, np.array(mean))

        self.model.save_lambda_network(self.folder_name, iteration)

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

    def optimize(self, interval_length, num_intervals_, time_per_cycle, num_epochs=1000):
        """Carries out optimization of protocol
        Args:
            interval_length: Number of time steps between each step of optimization
            num_intervals_: The number of intervals for the cycle. This is used just to verify calculations
            were computed correctly
            time_per_cycle: The time length of each cycle (tau)
            num_epochs: An int representing the number of epochs to train for
        Saves:
            loss.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                      the loss at each interval of optimization
            var.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                     the variance of particles at each interval of optimization
            mean.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing 
                      the mean of particles at each interval of optimization
        """
        num_steps = len(self.var)
        time_per_interval = interval_length * self.dt
        num_intervals = int(time_per_cycle / time_per_interval)
        num_intervals_per_step = num_intervals//num_steps
        assert num_intervals % num_steps == 0
        assert num_intervals_ == num_intervals

        all_loss = []
        all_var = []
        all_mean = []
        # interval_length + 1 times but interval_length time steps
        # The actual values of t don't matter for integrations,
        # just the dt and number of time steps
        ts = torch.arange(0, time_per_interval + self.dt, self.dt)
        assert ts.shape[0] == (interval_length + 1)
        for epoch in range(num_epochs):
            init_pos = self._initialize_position()
            all_loss.append([])
            all_var.append([])
            all_mean.append([])
            epoch_folder_name = self.folder_name + str(epoch) + "/"
            os.system("mkdir " + epoch_folder_name)
            for step in range(num_steps):
                all_loss[-1].append([])
                all_mean[-1].append([])
                all_var[-1].append([])
                self.sde.set_protocol_step(step)
                for i in range(num_intervals_per_step):
                    print(epoch, step, i)
                    ts = ts.detach()  # This step is probably superfluous
                    t_frac = i / num_intervals_per_step
                    self.sde.set_t_frac(torch.tensor([t_frac],
                                                     device=self.device))
                    ys = torchsde.sdeint(self.sde, init_pos, ts,
                                         dt=self.dt, method='euler')
                    mean_target, var_target = self._get_mean_var(t_frac,
                                                                 var_init=self.var[step],
                                                                 var_final=self.var[(step + 1) % num_steps])

                    init_pos = ys[-1].detach()
                    mean_count = ys[-1].mean()
                    var_count = ys[-1].var()
                    l = (nn.MSELoss()(mean_count,
                                      mean_target)
                         + nn.MSELoss()(var_count,
                                        var_target))
                    all_loss[-1][-1].append(l.item())
                    all_mean[-1][-1].append(mean_count.item())
                    all_var[-1][-1].append(var_count.item())
                    self.model.lambda_optimizer.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        l.backward()
                    nn.utils.clip_grad_norm_(self.sde.lambda_.parameters(), 1)
                    self.model.lambda_optimizer.step()

                    tag = ("epoch_" + str(epoch) + "_pstep_"
                           + str(step) + "_tfrac_" + str(i))

                    # self.plot_statistics(ys[-1].detach().flatten(),
                    #  epoch_folder_name, tag)
                    del ys, mean_count, var_count, l

                self._save_training_data(epoch, all_loss, all_var, all_mean)

    def test(self, interval_length, num_intervals_, time_per_cycle, num_iterations=1000):
        """Carries out num_iterations iterations of current protocol
        Args:
            interval_length: Number of time steps between each step of optimization
            num_intervals_: The number of intervals for the cycle. This is used just to verify calculations
            were computed correctly
            time_per_cycle: The time length of each cycle (tau)
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
            test_energy.npy: A numpy array of shape (num_epochs) representing the total 
                             energy produced per cycle
            test_work.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing
                            the total work produced at each interval
            test_dissipation.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step) representing
                                  the total entropy prduction at each interval
            str(epoch) + _all_y.pt: A numpy array of shape (num_steps, num_intervals_per_step) representing 
                                    the current positions

        """

        num_steps = len(self.var)
        time_per_interval = interval_length * self.dt
        num_intervals = int(time_per_cycle / time_per_interval)
        num_intervals_per_step = num_intervals//num_steps
        assert num_intervals % num_steps == 0
        assert num_intervals_ == num_intervals

        init_pos = self._initialize_position()

        all_loss = []
        all_dissipation = []
        all_work = []
        all_heat = []
        all_energy = []
        all_mean = []
        all_var = []

        #Get the temperature and k at the start of the protocol
        self.sde.set_protocol_step(0)
        self.sde.set_t_frac(torch.tensor([0.],
                                         device=self.device))

        init_temp, init_k = self.sde.get_T_k()

        # Temporarily set the initial protocol to be the last [T, k]
        # Only matters for work computation for the first interval of the first epoch
        self.sde.set_protocol_step(num_steps - 1)
        self.sde.set_t_frac(torch.tensor([(num_intervals_per_step - 1) / num_intervals_per_step],
                                         device=self.device))

        temp, k = self.sde.get_T_k()

        # interval_length + 1 times but interval_length time steps
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
            init_cycle_pos = init_pos.detach().clone()
            for step in range(num_steps):
                all_loss[-1].append([])
                all_dissipation[-1].append([])
                all_work[-1].append([])
                all_heat[-1].append([])
                all_mean[-1].append([])
                all_var[-1].append([])
                all_y.append([])
                self.sde.set_protocol_step(step)
                for i in range(num_intervals_per_step):
                    ts = ts.detach()  # This step is probably superfluous
                    t_frac = i / num_intervals_per_step
                    self.sde.set_t_frac(torch.tensor([t_frac],
                                                     device=self.device))

                    ys = torchsde.sdeint(self.sde, init_pos, ts,
                                         dt=self.dt, method='euler')

                    mean_target, var_target = self._get_mean_var(t_frac,
                                                                 var_init=self.var[step],
                                                                 var_final=self.var[(step + 1) % num_steps])

                    init_pos = ys[-1].detach()
                    #Use -2 because -1 represents the position at the start of the next interval
                    #0..-2 is one interval
                    mean_count = ys[-2].mean()
                    var_count = ys[-2].var()

                    all_mean[-1][-1].append(mean_count.item())
                    all_var[-1][-1].append(var_count.item())
                    all_y[-1].append(init_pos.tolist())

                    l = (nn.MSELoss()(mean_count,
                                      mean_target)
                         + nn.MSELoss()(var_count,
                                        var_target))
                    all_loss[-1][-1].append(l.item())

                    old_temp, old_k = temp, k
                    temp, k = self.sde.get_T_k()
                    dissipation = compute_dissipation(ys, temp, k)
                    work = compute_work(ys[0], k_i=old_k, k_f=k)
                    heat = compute_heat(ys, k)

                    all_dissipation[-1][-1].append(dissipation)
                    all_work[-1][-1].append(work)
                    all_heat[-1][-1].append(heat)
                    del ys, mean_count, var_count, l
            torch.save(all_y, str(epoch) + "_all_y.pt")
            # Computing the change in energy for k at the very end of the cycle
            energy_change = compute_energy(init_k, init_cycle_pos, init_pos)
            all_energy.append(energy_change)
            self._save_test_data(all_loss, all_dissipation, all_work,
                                 all_heat, all_energy, all_mean, all_var)
