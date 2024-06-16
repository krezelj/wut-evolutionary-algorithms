import json

import numpy as np
import gymnasium as gym
import pandas as pd

from ea.evolution import Evolution
from agent import Agent


def run_experiment(config_path: str = './ea/config.json', verbose: int = 0):
    with open(config_path, 'r') as f:
        config = json.load(f)

    env = gym.make(config["env_name"])
    Agent.architecture = [6, 6, 6, 3]
    Agent.ENV = env
    Agent.SEED = 1
    np.random.seed(0)

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
        best, fitness = evolution.simulate_one_generation()
        fitness *= 10
        if verbose > 0:
            print(f"finished ({fitness})")
        if fitness == 50.0:
            break
    
    env = gym.make(config["env_name"], render_mode="human")
    Agent.ENV = env
    for i in range(5):
        print(best.evaluate())


def main():
    run_experiment()


if __name__ == '__main__':
    pass