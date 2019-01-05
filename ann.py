import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output
from sklearn.datasets import make_circles

# CLASE PARA LAS CAPAS DE UNA ANN
class NeuralLayer():

	# n_conn: número de conexiones de las neuronas
	# n_neur: número de neuronas
	# act_f: función de activación
	def __init__(self, n_conn, n_neur, act_f):

		self.act_f=act_f

		# vector b: Bias, tantos como neuronas, es decir b=n_neur
		# random values: (-1, 1)
		self.b = np.random.rand(1, n_neur) * 2 -1

		# matriz w [n_conn x n_neur]: pesos, ej: 3 conexiones => 1 neurona
		self.w = np.random.rand(n_conn, n_neur) * 2 -1

# Crear TOPOLOGÍA DE LA ANN
def create_nn(topology, act_f):
	
	# CREACIÓN DE LA ANN, VECTOR DE CAPAS
	neural_net = []

	for l, layer in enumerate(topology[: -1]):
		# Inserta las capas a la ANN
		# índice l: número de capas
		neural_net.append(NeuralLayer(topology[l], topology[l+1], act_f))

	return neural_net

# FUNCIÓN DE ENTRENAMIENTO
# X: datos entrada
# Y: datos salida esperada
def train(neural_net, X, Y, e2medio, learning_rate=0.5, train=True):
	# X, Y: matrices
	
	# vector output: salida
	output = [(None, X)]

	# Forward pass
	print("\n >> think1: Forward pass")
	for l, layer in enumerate(neural_net):
		# l: recorre las capas de la neural_net
		# z: suma ponderada
		print("\n > Capa", l)
		print("entrada por capa")
		print(output[-1][1])
		print("neural_net[l].w")
		print(neural_net[l].w)
		print("neural_net[l].b")
		print(neural_net[l].b)
		print("entrada @ weights")
		print(output[-1][1] @ neural_net[l].w)
		print("ponderación + bias")
		z = output[-1][1] @ neural_net[l].w + neural_net[l].b
		print(z)

		# a: salida capa1
		print("activación")
		a = neural_net[l].act_f[0](z)
		print(a)

		output.append((z, a))

	if train:
		# Backward pass
		# Backpropagation algorithm

		print("\n >> backpropagation")
		deltas = []
		# len(neural_net): número de capas de la ANN
		for l in reversed(range(0, len(neural_net))):
			print("\n > Capa", l)
			# output[l+1][0]: suma ponderada
			z = output[l+1][0]
			# output[l+1][1]: activación
			a = output[l+1][1]
			if l == len(neural_net) -1:
				# Calcular delta última capa con respecto al coste
				print("*cálculo deltas última capa")
				print("activación")
				print(a)
				print("salida esperada")
				print(Y)
				print("derivada del ECM a - Y")
				print(e2medio[1](a, Y))
				print("derivada función de activación")
				print(neural_net[l].act_f[1](a))
				print("delta = derivada ECM * derivada función activación")
				deltas.insert(0, e2medio[1](a, Y) * neural_net[l].act_f[1](a))
				print(deltas)
			else:
				# Calcular capa respecto de capa previa
				print("cálculo deltas anteriores")
				print("activación")
				print(a)
				print("deltas")
				print(deltas[0])
				print("weights, capa ",l+1)
				print(_w)
				print("weights.T")
				print(_w.T)
				print("activación")
				print(a)
				print("derivada función activación")
				print(neural_net[l].act_f[1](a))
				print("deltas @ weights.T")
				print(deltas[0] @ _w.T)
				print("nueva delta: (deltas @ weights.T) * derivada función de activación")
				deltas.insert(0, deltas[0] @ _w.T * neural_net[l].act_f[1](a))
				print(deltas)

			_w = neural_net[l].w

			# Gradiente descendiente
			print("\n > Gradiente descendiente")
			print("**actualización de bias**")
			print("bias")
			print(neural_net[l].b)
			print("promedio de la delta")
			print(np.mean(deltas[0], axis=0, keepdims=True))
			print("learning_rate: ",learning_rate)
			print("error = promedio de deltas * learning_rate")
			print(np.mean(deltas[0], axis=0, keepdims=True) * learning_rate)
			print("bias - error")
			print(neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate)
			neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
			print("**actualización de weights**")
			print("weights")
			print(neural_net[l].w)
			print("output, capa anterior")
			print(output[l][1])
			print("output.T")
			print(output[l][1].T)
			print("delta, capa ",l)
			print(deltas[0])
			print("learning_rate: ",learning_rate)
			print("output.T @ deltas")
			print(output[l][1].T @ deltas[0])
			print(" error (output.T @ deltas) * learning_rate")
			print(output[l][1].T @ deltas[0] * learning_rate)
			print("weights - error")
			print(neural_net[l].w - output[l][1].T @ deltas[0] * learning_rate)
			neural_net[l].w = neural_net[l].w - output[l][1].T @ deltas[0] * learning_rate

	return output[-1][1]

# Función de activación
# SIGMOIDAL
# sigmodial[0](x) // función de activación Sigmoidal
# sigmodial[1](x) // derivada de la Sigmoidal
sigmoidal = (lambda x: 1 / (1 + np.e ** (-x)),
			lambda x: x * (1 - x))

# Función de costo
# ERROR CUADRÁTICO MEDIO
# yp: salida real de la ANN
# yr: salida predicha de la entrada
# e2medio[0](yp, yr) // función
# e2medio[1](yp, yr) // derivada
e2medio = (lambda yp, yr: np.mean((yp, yr)) ** 2,
			lambda yp, yr: (yp - yr))

if __name__ == "__main__":
	# Número de registros (filas)
	n = 4
	# Características (columnas)
	p = 2

	topology = [p, 2, 3, 1]

	# hacer print a X y Y para visualizar datos
	# matriz X: entradas
	# matriz Y: salidas
	X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

	Y = Y[:, np.newaxis]

	print("Entradas")
	print(X)
	print("\nSalidas esperadas")
	print(Y)

	# vector neural_net: vector de capas
	neural_net = create_nn(topology, sigmoidal)

	print("\nRandom starting synaptic weights:\n")
	print("Layer1:")
	print(neural_net[0].w)
	print("Layer2:")
	print(neural_net[1].w)
	print("Layer3:")
	print(neural_net[2].w)

	loss = []

	#X = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
	#Y = np.array([0,1,1,0])
	# PASAR A FUNCIÓN
	for i in range(1):
		out = train(neural_net, X, Y, e2medio, learning_rate=0.05)
		#print("out")
		#print(out)
		#if i % 250 == 0:
			#print(out)
			#loss.append(e2medio[0](out, Y))
			#res = 50

			#_x0 = np.linspace(-1.5, 1.5, res)
			#_x1 = np.linspace(-1.5, 1.5, res)

			#_Y = np.zeros((res, res))

			#for i0, x0 in enumerate(_x0):
				#for i1, x1 in enumerate(_x1):
					#_Y[i0, i1] = train(neural_net, np.array([[x0,x1]]), Y, e2medio, train=False)[0][0]

			#plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
			#plt.axis("equal")

			#plt.scatter(X[ Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
			#plt.scatter(X[ Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

			#clear_output(wait=True)
			#plt.show()
			#plt.plot(range(len(loss)), loss)
			#plt.show()
			#time.sleep(0.5)

	print("\nNew synaptic weights after training:\n")
	print("Input layer, 2 neurons => Hidden Layer1, 2 neurons:")
	print(neural_net[0].w)
	print("\nHidden Layer1, 2 neurons => Output Layer, 1 neuron:")
	print(neural_net[1].w)