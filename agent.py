from typing import Optional
import gymnasium as gym
import numpy as np
import numpy.typing as npt

Weights = list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]

class Agent:

    RUNS_PER_EVALUATION = 10
    SEED = 0
    ENV: gym.Env = None
    
    architecture: list[int] = [4, 4, 2]
    hidden_activation = lambda _, x : np.clip(x, a_min=0, a_max=None) # relu
    output_activation = lambda _, x : x

    def __init__(self, genes: Optional[Weights] = None) -> None:
        if genes is None:
            self.__init_weights()
        else:
            self.genes = genes
    
    def evaluate(self) -> float:
        total_reward = 0
        for _ in range(Agent.RUNS_PER_EVALUATION):
            state, _ = self.ENV.reset(seed=self.SEED)
            self.SEED += 1
            done = False
            while not done:
                action = self.__get_action(state)
                state, reward, done, truncated, _ = self.ENV.step(action)
                done = done or truncated
                total_reward += reward
        return total_reward / (Agent.RUNS_PER_EVALUATION * 10.0)

    def mutate(self, prob: float = 1.0, strength: float = 0.02) -> None:
        for W, b in self.genes:
            dW = (np.random.uniform(size=W.shape) <= prob) * np.random.normal(loc=0, scale=strength, size=W.shape)
            db = (np.random.uniform(size=b.shape) <= prob) * np.random.normal(loc=0, scale=strength, size=b.shape)
            W += dW
            b += db

    def copy(self) -> 'Agent':
        genes_copy = []
        for W, b in self.genes:
            genes_copy.append((
                W.copy(),
                b.copy()
            ))
        return Agent(genes_copy)
    
    def __init_weights(self):
        self.genes = []
        for i in range(len(self.architecture) - 1):
            self.genes.append((
                np.random.uniform(-1, 1, size=(self.architecture[i + 1], self.architecture[i])),
                np.random.uniform(-1, 1, size=(self.architecture[i + 1], 1))
            ))

    def __get_action(self, state: npt.NDArray):
        output = state.reshape((self.architecture[0], 1))
        for i, (W, b) in enumerate(self.genes):
            if i < len(self.genes) - 1:
                output = self.hidden_activation(W @ output + b)
            else:
                output = self.output_activation(W @ output + b)
        return np.argmax(output)