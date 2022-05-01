import argparse
import sys
import torch
from directoptimization import DirectOptimization

sys.setrecursionlimit(160000)
folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--num_intervals", type=int, default=1000)
# number of steps before each gradient update
parser.add_argument("--interval_length", type=int, default=10)
parser.add_argument("--reg", type=float, default=0)
parser.add_argument("--tau", type=float, default=1.)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--clip_gradients", action='store_true', default=False)


config = parser.parse_args()

interval_length = config.interval_length
tau = config.tau
dt = config.dt
clip_gradients = config.clip_gradients
reg = config.reg
num_intervals = config.num_intervals
end_point_densities = torch.load("/scratch/users/shriramc/040922QubitProtocolTesting/tau_500_end_point_densities.pt")
end_point_densities = end_point_densities.real

if (dt * interval_length > tau):
    raise ValueError("Increase the Time Per Cycle")


d = DirectOptimization(end_point_densities=end_point_densities,
                       tau=tau,
                       dt=dt, folder_name=folder_name)
d.model.load_lambda_network(folder_name=folder_name)
if tau < 0.5:
    burn_in_steps=3
else:
    burn_in_steps=1
d.test(interval_length=interval_length,
           num_intervals_=num_intervals,
           burn_in_steps=burn_in_steps)
