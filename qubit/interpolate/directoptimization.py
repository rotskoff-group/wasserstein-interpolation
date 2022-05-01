import torch
import torch.nn as nn
import numpy as np
from simulate import Simulate
from networks import Model


class DirectOptimization:
    def __init__(self, end_point_densities, tau,
                 dt=0.001, device=torch.device("cpu"),
                 model=None, folder_name="./"):
        """Args:
            tau: A float representing the time of the cycle
            end_point_densities: A tensor of shape (num_steps + 1, num_steps) representing the densities at the beginning of each step
                                 The density at the last index should be the same as the density at the first index.
            model: The current model. If None creates a new model
            dt: The time step of integration
        """
        self.device = device
        self.tau = tau
        self.end_point_densities = end_point_densities
        if model is None:
            end_point_protocol = [[1, 1.5],
                                  [1.5, 2],
                                  [2, 1.5],
                                  [1.5, 1],
                                  [1, 1.5]]
            self.model = Model(tau=self.tau, end_point_protocol=end_point_protocol,
                               device=self.device)
        else:
            self.model = model
        self.dt = dt
        self.simulate = Simulate(lambda_=self.model.lambda_,
                                 dt=self.dt,
                                 device=device)
        self.folder_name = folder_name

    def get_target_density(self, protocol_step, t_frac):
        """
        Args:
            protocol_step: An int representing the current protocol step
            t_frac: A float representing the fraction of the current protocol step traversed (t/(tau) % (tau/num_steps))
        Returns:
            target_density: A tensor of shape (2) representing the target density at the current time
        """
        init_density = self.end_point_densities[protocol_step]
        final_density = self.end_point_densities[(protocol_step + 1)]
        target_density = final_density * t_frac + init_density * (1 - t_frac)
        return target_density

    def get_loss(self, curr_density, protocol_step, t_frac):
        """Gets MSE loss of the system
        Args:
            curr_density: A tensor of shape (2) representing the current density of the system
            protocol_step: An int representing the current protocol step
            t_frac: A float representing the fraction of the current protocol traversed (t/(tau) % (tau/num_steps))
        Returns:
            loss: The mean squared error loss of the current_density from the target_density
        """
        target_density = self.get_target_density(protocol_step=protocol_step,
                                                 t_frac=t_frac)
        curr_density = curr_density.diag().abs()
        loss = nn.MSELoss(reduction="sum")(target_density, curr_density)

        return loss

    def _save_training_data(self, iteration, loss):
        filename = self.folder_name + "loss.npy"
        np.save(filename, np.array(loss))
        self.model.save_lambda_network(self.folder_name, iteration)

    def optimize(self, interval_length, num_intervals_, num_epochs=1000,
                 burn_in_steps=1, reg=0, clip_gradients=True):
        """
        Args:
            interval_length: Number of time steps between each step of optimization
            num_intervals_: The number of intervals for the cycle. This is used just to verify calculations
                            were computed correctly
            num_epochs: An int representing the number of epochs to train for
            burn_in_steps: An int representing the number of burn in steps
        Saves:
            loss.npy: A numpy array of shape (num_epochs, num_steps, num_intervals_per_step)
                      representing the loss at each stage of optimization

        """
        num_steps = self.end_point_densities.shape[0] - 1
        time_per_interval = interval_length * self.dt
        num_intervals = int(self.tau / time_per_interval)
        num_intervals_per_step = num_intervals//num_steps
        time_per_step = self.tau / num_steps
        assert num_intervals % num_steps == 0
        assert num_intervals_ == num_intervals

        all_loss = []
        # interval_length time steps
        ts_base = torch.arange(0, time_per_interval, self.dt)
        assert ts_base.shape[0] == (interval_length)
        for epoch in range(num_epochs):
            print(epoch)
            # Initial state doesnt matter
            init_state = torch.rand((3))
            init_state /= (init_state.sum() * 2)
            init_density = None  # Not necessary to compute this
            all_loss.append([])
            for period in range(burn_in_steps + 1):
                # Burn in step(s) to equilibrate
                is_burn_in_stage = (period < burn_in_steps)
                for step in range(num_steps):
                    if not is_burn_in_stage:
                        all_loss[-1].append([])
                    for i in range(num_intervals_per_step):
                        t_init = i * time_per_interval + step * time_per_step
                        ts = (ts_base.detach() + t_init).unsqueeze(-1)
                        all_states, all_densities, grad_deviation = self.simulate.simulate(init_state=init_state,
                                                                                           init_density=init_density,
                                                                                           ts=ts,
                                                                                           interval_length=interval_length,
                                                                                           protocol_step=step,
                                                                                           track_gradients=not(is_burn_in_stage))
                        init_state = all_states[-1].detach()
                        init_density = all_densities[-1].detach()
                        if (is_burn_in_stage):
                            del all_states, all_densities
                            continue

                        t_frac = i / num_intervals_per_step

                        l = (self.get_loss(curr_density=all_densities[-2],
                                           protocol_step=step, t_frac=t_frac)

                             + reg * grad_deviation)

                        all_loss[-1][-1].append(l.item())

                        self.model.lambda_optimizer.zero_grad()
                        with torch.autograd.set_detect_anomaly(True):
                            l.backward()
                        if (clip_gradients):
                            nn.utils.clip_grad_norm_(self.model.lambda_.parameters(),
                                                     1)
                        self.model.lambda_optimizer.step()
                        del l, all_states, all_densities
            if (epoch % 25) == 0:
                self._save_training_data(epoch, all_loss)

    def test(self, interval_length, num_intervals_, burn_in_steps=1):
        """Carries out num_iterations iterations of current protocol
        Args:
            interval_length: Number of time steps between each step of optimization
            num_intervals_: The number of intervals for the cycle. This is used just to verify calculations
                            were computed correctly
            burn_in_steps: An int representing the number of burn in steps
        """

        num_steps = self.end_point_densities.shape[0] - 1
        time_per_interval = interval_length * self.dt
        num_intervals = int(self.tau / time_per_interval)
        num_intervals_per_step = num_intervals//num_steps
        time_per_step = self.tau / num_steps
        assert num_intervals % num_steps == 0
        assert num_intervals_ == num_intervals
        all_loss = []
        all_heat = []
        all_work = []
        all_dissipation = []
        all_states = []
        all_densities = []

        # interval_length time steps
        ts_base = torch.arange(0, time_per_interval, self.dt)
        assert ts_base.shape[0] == (interval_length)
        # Initial state doesnt matter
        init_state = torch.rand((3))
        init_state /= (init_state.sum() * 2)
        init_density = torch.randn((1, 2, 2))  # Not necessary to compute this


        for period in range(burn_in_steps + 1):
            # Burn in step(s) to equilibrate
            is_burn_in_stage = (period < burn_in_steps)
            for step in range(num_steps):
                if not is_burn_in_stage:
                    all_loss.append([])
                    all_heat.append([])
                    all_work.append([])
                    all_dissipation.append([])
                    all_states.append([])
                    all_densities.append([])
                for i in range(num_intervals_per_step):
                    t_init = i * time_per_interval + step * time_per_step
                    ts = (ts_base.detach() + t_init).unsqueeze(-1)
                    interval_states, interval_work, interval_heat, interval_dissipation, interval_densities = self.simulate.simulate_test(init_state=init_state,
                                                                                                                                          init_density=init_density,
                                                                                                                                          ts=ts,
                                                                                                                                          interval_length=interval_length,
                                                                                                                                          protocol_step=step)
                    init_state = interval_states[-1].detach()
                    init_density = interval_densities[-1].detach().unsqueeze(0)
                    if (is_burn_in_stage):
                        continue
                    t_frac = i / num_intervals_per_step
                    l = self.get_loss(curr_density=interval_densities[-2], protocol_step=step, t_frac=t_frac)
                    all_loss[-1].append(l.item())
                    all_states[-1].extend(interval_states[:-1])
                    all_densities[-1].extend(interval_densities[:-1])
                    all_work[-1].extend(interval_work)
                    all_heat[-1].extend(interval_heat)
                    all_dissipation[-1].extend(interval_dissipation)
                if not is_burn_in_stage:
                    all_states[-1] = torch.stack(all_states[-1], dim=0)
                    all_densities[-1] = torch.stack(all_densities[-1], dim=0)
                    all_work[-1] = torch.cat(all_work[-1], dim=0)
                    all_heat[-1] = torch.cat(all_heat[-1], dim=0)
                    all_dissipation[-1] = torch.cat(all_dissipation[-1], dim=0)


        all_states = torch.stack(all_states).cpu().numpy()
        all_densities = torch.stack(all_densities).cpu().numpy()
        all_work = torch.stack(all_work).cpu().numpy()
        all_heat = torch.stack(all_heat).cpu().numpy()
        all_dissipation = torch.stack(all_dissipation).cpu().numpy()

        filename = self.folder_name + "test_loss.npy"
        np.save(filename, np.array(all_loss))

        filename = self.folder_name + "test_states.npy"
        np.save(filename, all_states)
        filename = self.folder_name + "test_densities.npy"
        np.save(filename, all_densities)
        filename = self.folder_name + "test_work.npy"
        np.save(filename, all_work)
        filename = self.folder_name + "test_heat.npy"
        np.save(filename, all_heat)
        filename = self.folder_name + "test_dissipation.npy"
        np.save(filename, all_dissipation)
