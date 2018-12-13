from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# seed(): generador de números aleatorios, con una raíz
		# a fin de generar los mismos números aleatorios en cada proceso
		random.seed(1)

		# Modelo de la neurona:
		# 	Neuronas: 1
		# 	Conexiones de entrada: 3
		#	Conexiones de salida: 1

		# Asignamos los pesos aleatorios a una matriz de 3x1, con valores en el rango [-1,1]
		# y significa 0 ¿?
		self.synaptic_weights = 2 * random.random((3, 1)) - 1

	# Función sigmoidal
	def sigmoidal(self, x):
		return 1 / (1 + exp(-x))

	# Derivada de la función sigmoidal
	def derivada_sigmoidal(self, x):
		return x * (1 - x)

	### ENTRENAMIENTO DE LA RED NEURONAL ###
	def train(self, t_set_inputs, t_set_outputs, num_t_iterations):

		for iteration in range(num_t_iterations):
			# Entrenar la neurona
			# Output es una matriz 4x1
			output = self.think(t_set_inputs)

			# Cálculo de error
			# Error es una matriz 4x1
			error = t_set_outputs - output

			# Multiplica el error por la entrada y otra vez por el gradiente de Sigmoidal
			# Los pesos no confiables se ajustan
			# Las entradas que son cero no causan cambios en los pesos
			
			# dot(t_set_inputs.T, error) 3x1
			# dot(t_set_inputs.T, error * self.derivada_sigmoidal(output)) 3x1
			# self.derivada_sigmoidal(output)
			print("self.derivada_sigmoidal(output)\n")
			print(self.derivada_sigmoidal(output))
			# Adjustment es una matriz 3x1
			adjustment = dot(t_set_inputs.T, error * self.derivada_sigmoidal(output))

			# Ajuste de pesos
			# Suma de matrices
			self.synaptic_weights += adjustment

	# La red neuronal piensa
	# El return es una Matriz
	def think(self, inputs):
		# Pasa las entradas a través de la neurona
		# dot(inputs, self.synaptic_weights)) Matriz 4x1
		return self.sigmoidal(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

	# Inicializa red neuronal
	neural_net = NeuralNetwork()
	print("Random starting synaptic weights: ")
	print(neural_net.synaptic_weights)

	# Las pruebas.
	# Las entradas:
		# Ej1 | Ej2 | Ej3 | Ej4
		#  0  |  1  |  1  |  0
		#  0  |  1  |  0  |  1
		#  1  |  1  |  1  |  1

	# Las salidas:
		# S: Ej1=0 | Ej2=1 | Ej3=1 | Ej4=0
	
	# Las matrices
	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	# ENTRENAMIENTO
	neural_net.train(training_set_inputs, training_set_outputs, 1)

	# Pesos luego del entrenamiento
	print("New synaptic weights after training: ")
	print(neural_net.synaptic_weights)

	# Considerando una situación
	print("Considering new situation [1, 0, 0] -> ? ")
	print(neural_net.think(array([1, 0, 0])))

	print("Considering new situation [0, 0, 1] -> 0 ")
	print(neural_net.think(array([0, 0, 1])))

	print("Considering new situation [1, 1, 1] -> 1 ")
	print(neural_net.think(array([1, 1, 1])))

	print("Considering new situation [1, 0, 1] -> 1 ")
	print(neural_net.think(array([1, 0, 1])))

	print("Considering new situation [0, 1, 1] -> 0 ")
	print(neural_net.think(array([0, 1, 1])))

	print("Considering new situation [3, 1, 2] -> ? ")
	print(neural_net.think(array([3, 1, 2])))

	print("Considering new situation [0, 0, 0] -> ? ")
	print(neural_net.think(array([0, 0, 0])))