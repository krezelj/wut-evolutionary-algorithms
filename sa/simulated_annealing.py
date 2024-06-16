
from typing import List

import numpy as np

from cartpole_agent import CartPoleAgent


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

        # self.__init_first_generation()

    # def __init_first_generation(self):
    #     self.all_specimens : List[CartPoleAgent] = \
    #         np.array([CartPoleAgent(**self.init_args) for _ in range(self.generation_size)])
        
    def simulate(self, verbose: int = 0):
        current_specimen = CartPoleAgent()
        current_fitness = current_specimen.evaluate()

        best_specimen = current_specimen
        best_fitness = current_fitness

        while self.T > self.T_min:
            if verbose > 0:
                print(f"current temp: {self.T}, current fitness: {current_fitness}, best fitness: {best_fitness}")
            for _ in range(self.C):
                candidate = current_specimen.copy()
                candidate.mutate()

                candidate_fitness = candidate.evaluate()
                p = 1.0 if candidate_fitness > current_fitness else \
                    np.exp(-(current_fitness - candidate_fitness) / self.T)
                if np.random.random() < p:
                    current_specimen = candidate
                    current_fitness = candidate_fitness
                if current_fitness > best_fitness:
                    best_specimen = current_specimen
                    best_fitness = current_fitness

            self.T *= self.alpha

        return best_specimen
        
    # def __create_new_specimen(self, weights = None):
    #     if weights is None:
    #         weights = np.ones(self.generation_size)
    #     parent: CartPoleAgent = np.random.choice(self.all_specimens, size=1, replace=True, p=weights)[0]
    #     new_specimen : CartPoleAgent = parent.copy()
    #     new_specimen.mutate(**self.mutation_args)
    #     return new_specimen
    
    # def get_best_specimen(self) -> tuple[CartPoleAgent, float]:
    #     fitness = np.array([
    #         self.all_specimens[i].evaluate() for i in range(self.generation_size)
    #     ])
    #     return self.all_specimens[fitness.argmax()], fitness.max()


