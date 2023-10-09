# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
from solver import *
from instance.instance import Instance
from simulator.simulator import Simulator
import random

ran_seed = 123
np.random.seed(ran_seed)
random.seed(ran_seed)

if __name__ == "__main__":
    current_dir = os.getcwd()+"\EVRPTW-main-DBProduction"

    with open("EVRPTW-main-DBProduction\etc\settings.json", "r") as json_in:
        configs = json.load(json_in)

    with open("EVRPTW-main-DBProduction\etc\settings_solver.json", "r") as json_in:
        configs_solver = json.load(json_in)

    # base simulation with 1 run and configuration in configs
    configs["main_path"] = current_dir
    # configs_solver["consider_pickup"] = "Yes"
    # configs_solver["consider_partial_recharge"] = "Yes"
    simulator = Simulator(configs)
    
    instance = Instance(configs, simulator) #prende in input settings 

    solver = Solver(configs_solver, instance)
    solution, final_OF = solver.solve(ran_seed)
