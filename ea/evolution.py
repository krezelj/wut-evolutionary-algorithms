from cartpole_agent import CartPoleAgent
import numpy as np
from scipy.special import softmax
from typing import List

N_PARENTS_IF_CROSSOVER = 2

class Evolution():

    def __init__(self, 
                 generation_size: int,
                 top_k: int,
                 init_args: dict = {},
                 mutation_args: dict = {},
                ) -> None:
        assert(top_k <= generation_size)
        self.generation_size = generation_size
        self.init_args = init_args
        self.mutation_args = mutation_args
        self.top_k = top_k
        self.__init_first_generation()

    def __init_first_generation(self):
        self.all_specimens : List[CartPoleAgent] = \
            np.array([CartPoleAgent(**self.init_args) for _ in range(self.generation_size)])

    def simulate(self, n_generations):
        for i in range(n_generations):
            if (i+1) % 5 == 0:
                print(f'generation: {i+1}')
            self.simulate_one_generation()

    def simulate_one_generation(self, verbose: int = 0):
        new_specimens = np.empty(self.generation_size, dtype=CartPoleAgent)

        # evaluate all current specimen
        fitness = np.array([
            self.all_specimens[i].evaluate() for i in range(self.generation_size)
        ])
        if verbose > 0:
            print(f"best fitness this generation: {fitness.max()}")
        if np.all(np.isinf(fitness)):
            print("all fitness are -infinity")

        # hold out best
        top_idx = np.argpartition(fitness, -self.top_k)[-self.top_k:]
        new_specimens[:self.top_k] = self.all_specimens[top_idx]

        # create the remaining
        weights = softmax(fitness)
        for i in range(self.top_k, self.generation_size):
            new_specimens[i] = self.__create_new_specimen(weights)
        
        # update
        self.all_specimens = new_specimens

    def __create_new_specimen(self, weights = None):
        if weights is None:
            weights = np.ones(self.generation_size)
        parent: CartPoleAgent = np.random.choice(self.all_specimens, size=1, replace=True, p=weights)[0]
        new_specimen : CartPoleAgent = parent.copy()
        new_specimen.mutate(**self.mutation_args)
        return new_specimen
    
    def get_best_specimen(self) -> tuple[CartPoleAgent, float]:
        fitness = np.array([
            self.all_specimens[i].evaluate() for i in range(self.generation_size)
        ])
        return self.all_specimens[fitness.argmax()], fitness.max()
    

def main():
    pass

if __name__ == '__main__':
    main()