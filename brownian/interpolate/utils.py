from asyncore import compact_traceback
import torch


def compute_dissipation(traj, T, k):
    """Computes total entropy production in the medium 
       using Stratonovich convention for a constant temperature 
       and constant harmonic potential (i.e. k) trajectory of num_steps.
       It is necessary to include the step prior to the start of the 
       trajectory to compute heat.
    Args:
        traj: A torch tensor of shape (num_steps + 1, num_particles)
        T: A float representing the current temperature of the trajectory
        k: A float representing the current strength of the harmonic potential
    Returns:
        dissipation: A float reprsenting the total entropy production for the trajectory
    """
    heat = compute_heat(traj, k)
    dissipation = heat/T
    return dissipation


def compute_heat(traj, k):
    """Computes total heat production in the medium
        using Stratonovich convention for a constant
        harmonic potential (i.e. k) trajectory of num_steps. 
        It is necessary to include the step prior to the start of the 
        trajectory to compute heat.
    Args:
        traj: A torch tensor of shape (num_steps + 1, num_particles)
        k: A float representing the current strength of the harmonic potential
    Returns:
        heat: A float reprsenting the total heat production in the medium
    """
    heat = (0.5
            * ((-k * traj)[:-1] + (-k * traj)[1:])
            * torch.diff(traj, dim=0)).sum()
    heat = heat.item()
    return heat


def compute_work(positions, k_i, k_f):
    """Computes the work done on the system for a change in the 
       protocol (i.e. the value of k).
    Args:
        positions: A torch tensor of shape (num_particles)
        k_i: A float represting the strength of the harmonic potential before protocol change
        k_f: A float represting the strength of the harmonic potential after protocol change
    """
    work = (0.5
            * (k_f - k_i)
            * (positions ** 2)).sum()
    work = work.item()
    return work


def compute_energy(k, pos_i, pos_f):
    """Computes the total change in energy of a trajectory given positions at the start and end of
       the trajectory
    Args:
        k: A float represting the strength of the harmonic potential at a point in the trajectory
        pos_i: A torch tensor of shape (num_particles) representing the positions before the protocol is run
        pos_f: A torch tensor of shape (num_particles) representing the positions after the protocol is run
    Returns:
        del_e: A float representing the total change in energy of the cycle
    """
    del_e = (0.5
             * k
             * ((pos_i ** 2) - (pos_f ** 2))).sum()
    del_e = del_e.item()
    return del_e
