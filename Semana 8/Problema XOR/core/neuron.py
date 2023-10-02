import numpy as np

def escalonada(x):
    return 0 if x < 0 else 1

def sigmoide(x):
    return 1/(1 + np.exp(-x))


class Neuron:
    def __init__(self, inputs:list):
        self.inputs = inputs
        self.weights = [0]*len(inputs)
        self.b = 0
        self.funcion = None

    def calculate(self) -> float:
        if self.funcion == None:
            raise Exception("La funcion de activacion no ha sido definida")
        result = [self.inputs[i]*self.weights[i] for i in range(len(self.inputs))]
        result = sum(result) + self.b
        result = self.funcion(result)
        return result

    def set_weights(self, weights:list):
        self.weights = weights
        if len(self.inputs) != len(weights):
            raise Exception("Los pesos difieren de la cantidad de entradas")
    
    def set_umbral(self, umbral:float):
        self.b = umbral
    
    def set_activation_function(self, funcion):
        self.funcion = funcion
        
        

if __name__ == "__main__":
    inputs = [-1, 1]
    neurona = Neuron(inputs)

    neurona.set_activation_function(escalonada)
    print(neurona.calculate())
    neurona.set_weights([4, 5])
    neurona.set_umbral(-6)
    print(neurona.calculate())
