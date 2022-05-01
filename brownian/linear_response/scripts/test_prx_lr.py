import argparse
import sys

from pandas import interval_range
from prx_lr import Protocol

sys.setrecursionlimit(160000)
folder_name = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=float, default=1E-5)
parser.add_argument("--t_h", type=float, default=1.)
parser.add_argument("--t_c", type=float, default=0.1)
parser.add_argument("--var_h", type=float, default=0.1)
parser.add_argument("--var_l", type=float, default=0.01)
parser.add_argument("--num_intervals", type=int, default=1000)
parser.add_argument("--interval_length", type=int, default=1000)

parser.add_argument("--time_per_cycle", type=float, default=1.)
parser.add_argument("--dt", type=float, default=0.001)

config = parser.parse_args()

d = Protocol(d=config.d,
             t_h=config.t_h,
             t_c=config.t_c,
             var_h=config.var_h,
             var_l=config.var_l,
             num_intervals=config.num_intervals,
             dt=config.dt,
             folder_name=folder_name)

d.test(time_per_cycle=config.time_per_cycle,
       interval_length_=config.interval_length,
       num_iterations=1000)
