import numpy as np
from scipy.special import softmax

class Specimen():

    def __init__(self, genes) -> None:
        self.genes = genes

    def __str__(self) -> str:
        return "Specimen"
    
    def evaluate(self) -> float:
        return 0

    def mutate(self) -> None:
        return None    

    def copy(self) -> 'Specimen':
        return Specimen(self.genes)
    
    @classmethod
    def crossover(cls, *parents):
        new_genes = {}
        n_parents = len(parents)
        for k in parents[0].genes.keys():
            parent_idx = np.random.randint(0, n_parents)
            new_genes[k] = parents[parent_idx].genes[k]
        return new_genes
    
class NeuralNetwork(Specimen):

    def __init__(self, genes) -> None:
        self.genes = genes

    def __str__(self) -> str:
        return "NeuralNetwork"
    
    def evaluate(self) -> float:
        pass

    def mutate(self) -> None:
        pass

    def copy(self) -> 'NeuralNetwork':
        return NeuralNetwork(self.genes)
    
    @classmethod
    def crossover(cls, *parents):
        pass # not used

class MLP(Specimen):

    X = None
    Y = None
    activation_function = None
    loss_function = None
    output_activation = None

    def __init__(self, genes=None, architecture=None) -> None:
        if genes is None:
            if architecture is None: 
                raise ValueError
            self.genes = []
            for i in range(len(architecture) - 1):
                self.genes.append((
                    np.random.uniform(-1, 1, size=(architecture[i+1], architecture[i])),
                    np.random.uniform(-1, 1, size=(architecture[i+1], 1))
                ))
        else:
            self.genes = genes

    @classmethod
    def set_parameters(cls, X, Y, activation_function, output_activation, loss_function):
        cls.X = X
        cls.Y = Y
        cls.activation_function = activation_function
        cls.loss_function = loss_function
        cls.output_activation = output_activation

    def evaluate(self):
        output = self.X
        for i, (W, b) in enumerate(self.genes):
            if i < len(self.genes) - 1:
                output = MLP.activation_function(W @ output + b)
            else:
                output = MLP.output_activation(W @ output + b)
        return -MLP.loss_function(self.Y.T, output.T)

    def mutate(self, p=0.01, strength=0.2):
        for W, b in self.genes:
            dW = (np.random.uniform(size=W.shape) < p) * np.random.uniform(-strength/2, strength/2, size=W.shape)
            db = (np.random.uniform(size=b.shape) < p) * np.random.uniform(-strength/2, strength/2, size=b.shape)
            W += dW
            b += db
    
    def copy(self):
        raise NotImplementedError
    
    @classmethod
    def crossover(cls, *parents):
        new_genes = []
        if len(parents) == 1:
            for i in range(len(parents[0].genes)):
                new_genes.append((
                    np.copy(parents[0].genes[i][0]),
                    np.copy(parents[0].genes[i][1])
                ))    
            return MLP(new_genes)
        # assuming 2 parents
        
        for i in range(len(parents[0].genes)):
            W1, b1 = parents[0].genes[i]
            W2, b2 = parents[1].genes[i]

            pW = np.random.uniform(size=W1.shape) < 0.5
            pb = np.random.uniform(size=b1.shape) < 0.5
            new_genes.append((W1 * pW + W2 * (1-pW), b1 * pb + b2 * (1-pb)))
        return MLP(new_genes)
        

        


