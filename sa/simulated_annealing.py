
from typing import List

import numpy as np

from agent import Agent


class SimulatedAnnealing:

    def __init__(self,
                 T: float,
                 T_min: float,
                 alpha: float,
                 C: int,
                 init_args: dict = {},
                 mutation_args: dict = {},
                 ) -> None:
        self.init_args = init_args
        self.mutation_args = mutation_args

        self.T = T
        self.T_min = T_min
        self.alpha = alpha
        self.C = C

        self.current_specimen = Agent()
        self.current_fitness = self.current_specimen.evaluate()

        self.best_specimen = self.current_specimen
        self.best_fitness = self.current_fitness
        
    def simulate(self, verbose: int = 0):
        done = False
        while not done:
            _, _, done = self.simulate_one_iteration(verbose)
        return self.best_specimen

    def simulate_one_iteration(self, verbose: int = 0):
        if self.T < self.T_min:
            return self.best_specimen, self.best_fitness, True
        
        if verbose > 0:
            print(f"current temp: {self.T:.2f} ({self.current_fitness * 10}/{self.best_fitness * 10})")

        for _ in range(self.C):
            candidate = self.current_specimen.copy()
            candidate.mutate()

            candidate_fitness = candidate.evaluate()
            p = 1.0 if candidate_fitness > self.current_fitness else \
                np.exp(-(self.current_fitness - candidate_fitness) / self.T)
            if np.random.random() < p:
                self.current_specimen = candidate
                self.current_fitness = candidate_fitness
            if self.current_fitness > self.best_fitness:
                self.best_specimen = self.current_specimen
                self.best_fitness = self.current_fitness

        self.T *= self.alpha
        return self.best_specimen, self.best_fitness, False
    