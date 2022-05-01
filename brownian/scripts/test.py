import argparse
import sys
from directoptimization import DirectOptimization

sys.setrecursionlimit(160000)
folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--t_h", type=float, default=1.)
parser.add_argument("--t_c", type=float, default=0.1)
parser.add_argument("--var_h", type=float, default=0.1)
parser.add_argument("--var_l", type=float, default=0.01)
parser.add_argument("--num_intervals", type=int, default=1000)
# number of steps before each gradient update
parser.add_argument("--interval_length", type=int, default=10)

parser.add_argument("--time_per_cycle", type=float, default=1.)
parser.add_argument("--dt", type=float, default=0.001)

config = parser.parse_args()

d = DirectOptimization(t_h=config.t_h,
                       t_c=config.t_c,
                       var_h=config.var_h,
                       var_l=config.var_l,
                       dt=config.dt,
                       folder_name=folder_name)

d.model.load_lambda_network(folder_name=folder_name)
d.test(interval_length=config.interval_length,
       num_intervals_=config.num_intervals,
       time_per_cycle=config.time_per_cycle)
