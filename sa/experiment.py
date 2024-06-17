import json
import time

import numpy as np
import gymnasium as gym
import pandas as pd

from sa.simulated_annealing import SimulatedAnnealing
from agent import Agent


def run_experiment(config_path: str = './sa/config.json', verbose: int = 0):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data = pd.DataFrame(columns=['run', 'iteration', 'time_elapsed', 'best_fitness'])

    env = gym.make(config["env_name"])
    Agent.architecture = [4, 4, 2]
    Agent.ENV = env
    
    for run in range(config["n_runs"]):
        Agent.SEED = config["seed"] + run
        np.random.seed(config["seed"] + run)

        sa = SimulatedAnnealing(
            T=config["T"],
            T_min=config["T_min"],
            alpha=config["alpha"],
            C=config["C"],
            mutation_args=dict(
                prob=config["prob"],
                strength=config["strength"]
            )    
        )

        iteration = 0
        done = False
        while not done:
            start = time.time_ns()
            best, fitness, done = sa.simulate_one_iteration(verbose)
            duration = time.time_ns() - start
            fitness *= 10
            data.loc[len(data)] = [run, iteration, duration, fitness]
            iteration += 1
            if fitness >= config["fitness_threshold"]:
                break
    
    data.to_csv(config["output_path"] + config["config_name"] + ".csv")
    

def main():
    run_experiment()


if __name__ == '__main__':
    pass