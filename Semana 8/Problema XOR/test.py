import random

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        y = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.step_function(y)

    def train(self, training_data, num_epochs):
        for _ in range(num_epochs):
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
    
    def print_weights(self):
        print(self.weights)

# Datos de entrada y salidas deseadas
inputs = [
    [0, 0],
    [0, 0.5],
    [0.5, 0.5],
    [0.7, 0.8],
    [0.9, 0.95],
    [0.4, 0.3]
]
outputs = [0, 0, 0, 1, 1, 0]

# Preparar los datos de entrenamiento
training_data = list(zip(inputs, outputs))

# Crear un perceptrón y entrenarlo
perceptron = Perceptron(num_inputs=2)
perceptron.train(training_data, num_epochs=100)

# Probar el perceptrón entrenado
for inputs, _ in training_data:
    prediction = perceptron.predict(inputs)
    print(f"Entrada: {inputs}, Predicción: {prediction}")

