import json
import time

import numpy as np
import gymnasium as gym
import pandas as pd

from ea.evolution import Evolution
from agent import Agent


def run_experiment(config_path: str = './ea/config.json', verbose: int = 0):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data = pd.DataFrame(columns=['generation', 'time_elapsed', 'best_fitness'])

    env = gym.make(config["env_name"])
    Agent.architecture = [4, 4, 2]
    Agent.ENV = env
    Agent.SEED = config["seed"]
    np.random.seed(config["seed"])

    evolution = Evolution(
        generation_size=config["generation_size"],
        top_k=config["top_k"],
        mutation_args=dict(
            prob=config["prob"],
            strength=config["strength"]
        )
    )
    for generation in range(config["n_generations"]):
        if verbose > 0:
            print(f"starting generation {generation + 1}... ", end="")

        start = time.time_ns()
        best, fitness = evolution.simulate_one_generation()
        duration = time.time_ns() - start
        fitness *= 10
        data.loc[generation] = [generation, duration, fitness]

        if verbose > 0:
            print(f"finished ({fitness})")
        if fitness >= config["fitness_threshold"]:
            break
    
    data.to_csv(config["output_path"])
    

def main():
    run_experiment()


if __name__ == '__main__':
    pass