#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

// CLASE DE UNA CAPA DE LA RED
// Una capa de la red neuronal consta de:
//	- Neuronas: las neuronas actuales de la capa y de la capa anterior hacen una matriz
//	- Pesos: es una matriz. Las dimensiones de esta son:
//		> para las filas, el número de neuronas de la capa anterior; para las columnas, el número de neuronas actual
//	- Bias: el umbral de activación. También es una matriz. Las dimensiones de esta son:
//		> para las filas, 1; para las columnas, el número de neuronas de la actual capa
//	- Deltas: es una matriz. Resultado del cálculo de las derivadas de la función de activación respecto de las salidas de cada capa
class NeuralLayer{
	private:
		// Matriz de pesos
		float* weights;

		// Filas de la matriz de los pesos (neuronas de la capa anterior)
		int num_nant;

		// Columnas de la matriz de los pesos (neuronas de la capa actual)
		int num_neur;

		// Matriz de bias
		float* bias;

		// Matriz de deltas
		float* deltas;

		// Filas de las deltas
		int filas_deltas;

		// Columnas de las deltas
		int columnas_deltas;

	public:
		void setWeights(float* weights);
		void setNum_Nant(int num_nant);
		void setNum_Neur(int num_neur);
		void setBias(float* bias);
		void setDeltas(float* deltas);
		void setFilas_Deltas(int filas_deltas);
		void setColumnas_Deltas(int columnas_deltas);

		float* getWeights();
		int getNum_Nant();
		int getNum_Neur();
		float* getBias();
		float* getDeltas();
		int getFilas_Deltas();
		int getColumnas_Deltas();
};

// DELCARACIÓN DE LA ESTRUCTURA DE DATOS CORRESPONDIENTE A LA SALIDA DE CADA CAPA
struct Output{
	float* mat;
	int f;
	int c;
};

// DECLARACIÓN DE LAS FUNCIONES GLOBALES

// Creación de una capa de red
// Parámetros: neuronas de la capa anterior (num_nant), neuronas de la capa actual (num_neur)
NeuralLayer createNeuralLayer(int num_nant, int num_neur);

// Crea una matriz con números aleatorios
// Parámetros: filas y columnas de la matriz
float* createMatrizRandom(int filas, int columnas);

// Muestra en pantalla una matriz
// Parámetros: la matriz, y sus filas y columnas
void mostrarMatriz(float ar[], int filas, int columnas);

// Multiplicación matricial
// Multiplicación de dos matrices con dimensiones aleatorias, ara[] y arb[].
// Las dimensiones de la resultante son: filas (tomadas de las filas de ara[]) y columnas (tomadas de las columnas de arb[])
// Si las columnas de ara[] son diferentes a las filas de arb[], el cálculo no está permitido, esto quiere decir, que deben ser iguales 
//		(columnas de ara[] == filas de arb[])
// Parámetros: primera matriz (ara[]), filas de ara[], columnas de ara[], segunda matriz (arb[]), columnas de arb[], matriz resultante (arr[])
void dot(float ara[], int filas_a, int columnas_a, float arb[], int columnas_b, float arr[]);

// Suma las Bias con la matriz ponderada con los Pesos
// Parámetros: la matriz ponderada, sus filas y columnas, la matriz de bias y una matriz vacía para almacenar el resultado
void sumaBias(float mat_ponderada[], int filas_mp, int columnas_mp, float mat_bias[], float res[]);

// Pensar
// Función que realiza el algoritmo de FEED-FORWARD de una red neuronal. Las fases de este algoritmo son:
//	- Ponderación de pesos y entradas por capa
//	- Sumatoria por neurona y suma con bias
//	- Función de activación
// Delvuelve una matriz con los resultados de salida que, al inicio de entrenamiento de la red, serán aleatorios y con
// mucho margen de error
// Parámetros: la matriz de entrada (set de entrada), sus filas y columnas, la matriz resultante, la red neuronal y cantidad de capas
void think1(float entrada[], int filas_entrada, int columnas_entrada, float res[], NeuralLayer neural_net[], int n_layers);

// Entrenar
// Función que realiza las siguientes funciones:
//	- Forward pass: lo mismo que think1(), pero con la recuperación de las resultantes de salida por capa, incluyendo la matriz de entrada
//	- Backpropagation: el algoritmo de retropropagación que incluye el algoritmo del Descenso del Gradiente para actualizar pesos y bias respecto del error
//		resultante de la capa de salida de la red
// Este último algoritmo se explica de la siguiente forma:
//	- Para actualizar los pesos y bias de la última capa:
//		> Necesario hacer el cálculo del error con la derivada de la función de costo
//		> Calcular las deltas de la capa: pasando por la derivada de la función de activación el resultado del cálculo del error
//		> Actualización de pesos y bias: al momento de actualizar pesos y bias, se debe tomar en cuenta las deltas de la capa de salida
//			y la salida de la capa anterior
//	- Para actualizar los pesos y bias de las capas anteriores
//		> Necesario calcular las deltas de la capa: tomando como referencia las deltas de la capa siguiente y la salida de la capa actual
//			calculando la derivada de la función de activación con respecto de la salida resultante de la capa actual
//		> Actulizaciób de pesos y bias: tomar las deltas de la capa ya generadas y los pesos y bias de la misma capa
// El proceso del backpropagation se realiza hasta el final:
//	- Cuando el algoritmo de retropropagación se encuentra al inicio de la red, la capa de entrada tiene como resultante la matriz de entrada, la cual será
//		ocupada para actualizar los pesos de la capa siguiente
void train1(float entrada[], int filas_entrada, int columnas_entrada, float salida[], float filas_salida, float columnas_salida, NeuralLayer neural_net[], int n_layers, float learning_rate);

// Función de activación: TANGENTE HIPERBÓLICA
// Parámetros: la matriz a activar, sus filas y columnas, y la matriz resultante
void tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]);

// Derivada de la función de activación: TANGETE HIPERBÓLICA
// Parámetros: la matriz a derivar, sus filas y columnas, y la matriz resultante
void derivada_tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]);

// Función de activación: SIGMOIDAL
// Parámetros: la matriz a activar, sus filas y columnas, y la matriz resultante
void sigmoidal(float ar[], int filas_a, int columnas_a, float arr[]);

// Función de activación: TANGENTE HIPERBÓLICA
// Parámetros: la matriz a derivar, sus filas y columnas, y la matriz resultante
void derivada_sigmoidal(float ar[], int filas, int columnas, float arr[]);

// Llamada a la función de activación
// Parámetros: matriz ponderada, sus filas y columnas, y la matriz resultante
void activationFunction(float weigthed_mat[], int filas, int columnas, float res[]);

// Llamada a la derivada de la función de activación
// Parámetros: matriz activada, sus filas y columnas, y la matriz resultante
void derivActivationFunction(float activated_mat[], int filas, int columnas, float res[]);

// Derivada de la función de costo: ERROR CUADRÁTICO MEDIO
// Parámetros: matriz de la salida de la red, sus filas y columnas, la matriz de salidas esperadas y la matriz resultante
void derivada_e2medio(float aa[], int filas, int columnas, float ab[], float arr[]);

// Multiplicación no matricial
// Parámetros: matrizA, sus filas y columnas, y matrizB
// Devuelve un apuntador a la matriz resultante
float* multiplicar(float aa[], int filas, int columnas, float ab[]);

// Transposición de una matriz
// Parámetros: la matriz a transponer, filas y columnas de la matriz a transponer, resultante de la matriz transpuesta
void transponerMatriz(float ar[], int filas, int columnas, float res[]);

// Promedio por columnas de una matriz
// Parámetros: una matriz, sus filas y columnas, y la matriz resultante
void mean(float aa[], int filas, int columnas, float arr[]);

// Multiplicación de matriz por número flotante
// Parámetros: la matriz a multiplicar, sus filas y columnas, la matriz resultante y el número flotante
void multmatfloat(float aa[], int filas, int columnas, float res[], float learning_rate);

int main(){

	// --- COMIENZA LA PARTE DONDE LOS DATOS PUEDEN SER MODIFICADOS A SU GUSTO --- //
	
	// DATOS SOBRE LAS MATRICES DE ENTRADA Y SALIDA
	int filas_X = 4;
	// columnas_X: además de ser el número de columnas de la matriz de entrada es el número de neuronas de la capa de entrada
	int columnas_X = 2;
	int filas_Y = 4;
	// columnas_X: además de ser el número de columnas de la matriz de salida es el número de neuronas de la capa de salida
	int columnas_Y = 1;

	// Capa de entrada
	float X[filas_X*columnas_X] = {0,0, 0,1, 1,0, 1,1};
	// Resultados esperados de salida
	float Y[filas_Y*columnas_Y] = {0,1,1,0};

	// Capa de entrada
	//float X[filas_X*columnas_X] = {1,.25,-0.5};
	// Resultados esperados de salida
	//float Y[filas_Y*columnas_Y] = {1,-1,0};

	// DATOS SOBRE LA TOPOLOGÍA DE LA RED NEURONAL
	
	// Número de capas
	int n_layers = 3;

	// Creación de la red neuronal
	NeuralLayer neural_net[n_layers-1];

	// Topología de red
	// Modificar de acuerdo a la cantidad de capas
	// Cada posición representa una capa, y cada valor representa el número de neuronas de la capa
	int nn_topology[n_layers] = {columnas_X, 3, columnas_Y};

	// DATOS SOBRE EL ENTRENAMIENTO
	float learning_rate = 0.2;
	long int iteraciones = 100000;

	// --- FINALIZA LA PARTE DONDE LOS DATOS PUEDEN SER MODIFICADOS A SU GUSTO --- //

	for(int layer = 0; layer < n_layers-1; layer++){
		neural_net[layer] = createNeuralLayer(nn_topology[layer], nn_topology[layer+1]);
	}

	cout<<"Random starting synaptic weights"<<endl;
	for(int layer = 0; layer < n_layers-1; layer++){
		cout<<"\n >> Layer "<<layer<<endl;
		cout<<"\n Weights"<<endl;
		mostrarMatriz(neural_net[layer].getWeights(), neural_net[layer].getNum_Nant(), neural_net[layer].getNum_Neur());
		cout<<"\n Bias"<<endl;
		mostrarMatriz(neural_net[layer].getBias(), 1, neural_net[layer].getNum_Neur());
	}

	// Procedimiento de PENSAR (think1)
	float res1[filas_Y*columnas_Y];
	think1(X, filas_X, columnas_X, res1, neural_net, n_layers-1);

	cout<<"\n > Resultante prueba think1:"<<endl;
	mostrarMatriz(res1, filas_Y, columnas_Y);

	// TRAINING
	cout<<"\n > TRAINING";
	for(int t=0;t<iteraciones;t++){
		if(t%1000==0){
			cout<<" .";
		}
		train1(X, filas_X, columnas_X, Y, filas_Y, columnas_Y, neural_net, n_layers-1, learning_rate);
	}
	cout<<"\n > END TRAINING";

	cout<<"\n\nSynaptic weights after training"<<endl;
	for(int layer = 0; layer < n_layers-1; layer++){
		cout<<"\n >> Layer "<<layer<<endl;
		cout<<"\n Weights"<<endl;
		mostrarMatriz(neural_net[layer].getWeights(), neural_net[layer].getNum_Nant(), neural_net[layer].getNum_Neur());
		cout<<"\n Bias"<<endl;
		mostrarMatriz(neural_net[layer].getBias(), 1, neural_net[layer].getNum_Neur());
	}

	// Procedimiento de PENSAR (think1)
	float res2[filas_Y*columnas_Y];
	think1(X, filas_X, columnas_X, res2, neural_net, n_layers-1);

	cout<<"\n > Resultante prueba think1:"<<endl;
	mostrarMatriz(res2, filas_Y, columnas_Y);
}

void NeuralLayer::setWeights(float* weights){
	this->weights = weights;
}

void NeuralLayer::setNum_Nant(int num_nant){
	this->num_nant = num_nant;
}

void NeuralLayer::setNum_Neur(int num_neur){
	this->num_neur = num_neur;
}

void NeuralLayer::setBias(float* bias){
	this->bias = bias;
}

void NeuralLayer::setDeltas(float* deltas){
	this->deltas = deltas;
}

void NeuralLayer::setFilas_Deltas(int filas_deltas){
	this->filas_deltas = filas_deltas;
}

void NeuralLayer::setColumnas_Deltas(int columnas_deltas){
	this->columnas_deltas = columnas_deltas;
}

float* NeuralLayer::getWeights(){
	return this->weights;
}

int NeuralLayer::getNum_Nant(){
	return this->num_nant;
}

int NeuralLayer::getNum_Neur(){
	return this->num_neur;
}

float* NeuralLayer::getBias(){
	return this->bias;
}

float* NeuralLayer::getDeltas(){
	return this->deltas;
}

int NeuralLayer::getFilas_Deltas(){
	return this->filas_deltas;
}

int NeuralLayer::getColumnas_Deltas(){
	return this->columnas_deltas;
}

NeuralLayer createNeuralLayer(int num_nant, int num_neur){
	NeuralLayer nl_temp;
	nl_temp.setWeights(createMatrizRandom(num_nant, num_neur));
	nl_temp.setNum_Nant(num_nant);
	nl_temp.setNum_Neur(num_neur);
	nl_temp.setBias(createMatrizRandom(1, num_neur));

	return nl_temp;
}

float* createMatrizRandom(int filas, int columnas){
	float* mat_temp = new float[filas * columnas];

	srand(time(NULL));
	for(int i = 0; i < filas; i++){
		for(int j = 0; j < columnas; j++){
			mat_temp[(i*columnas)+j] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
		}
	}

	return mat_temp;
}

void mostrarMatriz(float ar[], int filas, int columnas){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			cout<<ar[(i*columnas)+j]<<" ";
		}
		cout<<endl;
	}
}

void dot(float ara[], int filas_a, int columnas_a, float arb[], int columnas_b, float arr[]){
	for (int i = 0; i < filas_a; i++) {
        for (int j = 0; j < columnas_b; j++) {
            float sum = 0;
            for (int k = 0; k < columnas_a; k++) {
                sum = (ara[(i*columnas_a)+k] * arb[(k*columnas_b)+j])+sum;
            }
            arr[(i*columnas_b)+j] = sum;
        }
    }
}

void sumaBias(float mat_ponderada[], int filas_mp, int columnas_mp, float mat_bias[], float res[]){
	for(int i=0;i<filas_mp;i++){
		for(int j=0;j<columnas_mp;j++){
			res[(i*columnas_mp)+j]=mat_ponderada[(i*columnas_mp)+j]+mat_bias[j];
		}
	}
}

void tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]){
	float exp_value1;
	float exp_value2;

	for(int i=0;i<filas_a;i++){
		for(int j=0;j<columnas_a;j++){
			exp_value1=exp((double) ar[(i*columnas_a)+j]);
			exp_value2=exp((double) - ar[(i*columnas_a)+j]);

			arr[(i*columnas_a)+j] = (exp_value1-exp_value2)/(exp_value1+exp_value2);
		}
	}
}

void derivada_tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]){
	/*float resultante[filas_a*columnas_a];
	tang_hiper(ar, filas_a, columnas_a, resultante);*/

	for(int i=0;i<filas_a;i++){
		for(int j=0;j<columnas_a;j++){
			arr[(i*columnas_a)+j] = 1.0 - pow(ar[(i*columnas_a)+j],2);
		}
	}
}

void sigmoidal(float ar[], int filas_a, int columnas_a, float arr[]){
	float exp_value;
	for(unsigned i=0;i<filas_a;i++){
		for(unsigned j=0;j<columnas_a;j++){
			exp_value=exp((double) - ar[(i*columnas_a)+j]);
			arr[(i*columnas_a)+j]= (1/(1+exp_value));
		}
	}
}

void derivada_sigmoidal(float ar[], int filas, int columnas, float arr[]){
	for(unsigned i=0;i<filas;i++){
		for(unsigned j=0;j<columnas;j++){
			arr[(i*columnas)+j]= ar[(i*columnas)+j] * (1 - ar[(i*columnas)+j]);
		}
	}
}

void activationFunction(float weigthed_mat[], int filas, int columnas, float res[]){
	sigmoidal(weigthed_mat, filas, columnas, res);
}

void derivActivationFunction(float activated_mat[], int filas, int columnas, float res[]){
	derivada_sigmoidal(activated_mat, filas, columnas, res);
}

void derivada_e2medio(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]-ab[(i*columnas)+j];
		}
	}
}

float* multiplicar(float aa[], int filas, int columnas, float ab[]){
	float* temp = new float[filas*columnas];
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			temp[(i*columnas)+j]=aa[(i*columnas)+j]*ab[(i*columnas)+j];
		}
	}

	return temp;
}

void transponerMatriz(float ar[], int filas, int columnas, float res[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			res[(j*filas)+i]=ar[(i*columnas)+j];
		}
	}
}

void mean(float aa[], int filas, int columnas, float arr[]){
	// Sumar matriz por columnas
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			arr[j]=(aa[(i*columnas)+j])+arr[j];
		}
	}

	for(int j=0;j<columnas;j++){
		arr[j] = arr[j]/filas;
	}
}

void zeros(float a[], int filas, int columnas){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			a[(i*columnas)+j]=0;
		}
	}
}

void multmatfloat(float aa[], int filas, int columnas, float res[], float learning_rate){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			res[(i*columnas)+j]=aa[(i*columnas)+j]*learning_rate;
		}
	}
}

float* resta(float aa[], int filas, int columnas, float ab[]){
	float* temp = new float[filas * columnas];

	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			temp[(i*columnas)+j]=aa[(i*columnas)+j]-ab[(i*columnas)+j];
		}
	}

	return temp;
}

void think1(float entrada[], int filas_entrada, int columnas_entrada, float res[], NeuralLayer neural_net[], int n_layers){
	float* resultante_temp = entrada;
	int filas = filas_entrada;
	int columnas = columnas_entrada;

	for(int layer=0;layer<n_layers;layer++){
		float res_temp[filas * neural_net[layer].getNum_Neur()];
		dot(resultante_temp, filas, columnas, neural_net[layer].getWeights(), neural_net[layer].getNum_Neur(), res_temp);
		// Variable 'columnas' cambiará después de la multiplicación matricial
		columnas = neural_net[layer].getNum_Neur();

		sumaBias(res_temp, filas, columnas, neural_net[layer].getBias(), res_temp);

		activationFunction(res_temp, filas, columnas, res);

		resultante_temp = res;
	}
}

void train1(float entrada[], int filas_entrada, int columnas_entrada, float salida[], float filas_salida, float columnas_salida, NeuralLayer neural_net[], int n_layers, float learning_rate){
	Output* output = new Output[n_layers+1];

	output[0].mat = entrada;
	output[0].f = filas_entrada;
	output[0].c = columnas_entrada;

	// FEED-FORWARD
	for(int layer_o=1;layer_o<n_layers+1;layer_o++){
		float res_temp[output[layer_o-1].f * neural_net[layer_o-1].getNum_Neur()];
		dot(output[layer_o-1].mat, output[layer_o-1].f, output[layer_o-1].c, neural_net[layer_o-1].getWeights(), neural_net[layer_o-1].getNum_Neur(), res_temp);
		// Variable 'columnas' cambiará después de la multiplicación matricial
		output[layer_o].c = neural_net[layer_o-1].getNum_Neur();

		sumaBias(res_temp, output[layer_o-1].f, output[layer_o].c, neural_net[layer_o-1].getBias(), res_temp);

		output[layer_o].mat = new float[output[layer_o-1].f * output[layer_o].c];
		activationFunction(res_temp, output[layer_o-1].f, output[layer_o].c, output[layer_o].mat);
		output[layer_o].f=output[layer_o-1].f;
	}

	// BACKPROPAGATION
	int layer_o = n_layers;

	for(int layer = n_layers-1; layer>=0; layer--){
		if(layer==n_layers-1){
			// cálculo de deltas ÚLTIMA CAPA
			// CALCULAR EL ERROR
			float error[output[layer_o].f * output[layer_o].c];
			derivada_e2medio(output[layer_o].mat, output[layer_o].f, output[layer_o].c, salida, error);

			// DERIVADA FUNCTACT
			float der_fa[output[layer_o].f * output[layer_o].c];
			derivActivationFunction(output[layer_o].mat, output[layer_o].f, output[layer_o].c, der_fa);

			// MULTIPLICACIÓN error * der_fa
			// CREACIÓN DE DELTA
			neural_net[layer].setDeltas(multiplicar(error, output[layer_o].f, output[layer_o].c, der_fa));
			neural_net[layer].setFilas_Deltas(output[layer_o].f);
			neural_net[layer].setColumnas_Deltas(output[layer_o].c);
		}else{
			//PROCESO
			// RECUPERACIÓN DELTA SIGUIENTE (layer+1)
			float* deltas_temp = neural_net[layer+1].getDeltas();
			int fd_temp = neural_net[layer+1].getFilas_Deltas();
			int cd_temp = neural_net[layer+1].getColumnas_Deltas();

			// RECUPERACIÓN DE PESOS SIGUIENTE (layer+1)
			float* weights_temp = neural_net[layer+1].getWeights();
			int fw_temp = neural_net[layer+1].getNum_Nant();
			int cw_temp = neural_net[layer+1].getNum_Neur();

			// TRANSPOSICIÓN DE LA MATRIZ DE LOS PESOS A OCUPAR (layer+1)
			float weights_temp_T[cw_temp * fw_temp];
			transponerMatriz(weights_temp, fw_temp, cw_temp, weights_temp_T);

			// RECUPERACIÓN OUTPUT CAPA ACTUAL Y DERIVADA FUNCTACT
			float der_fa[output[layer_o].f * output[layer_o].c];
			derivActivationFunction(output[layer_o].mat, output[layer_o].f, output[layer_o].c, der_fa);

			// DELTAS[layer+1] @ WEIGHTS_TEMPT_T
			float res_dot[fd_temp * fw_temp];
			dot(deltas_temp, fd_temp, cd_temp, weights_temp_T, fw_temp, res_dot);

			// MULTIPLICACIÓN RES_DOT * DER_FA
			// CREACIÓN DE LAS DELTAS CAPA [layer]
			neural_net[layer].setDeltas(multiplicar(res_dot, fd_temp, fw_temp, der_fa));
			neural_net[layer].setFilas_Deltas(fd_temp);
			neural_net[layer].setColumnas_Deltas(fw_temp);
		}
		// actualización de pesos y bias
		// PROCESO
		// actualización bias
		// PROMEDIO DE DELTAS
		float res_prom[1 * neural_net[layer].getColumnas_Deltas()];
		zeros(res_prom, 1, neural_net[layer].getColumnas_Deltas());
		mean(neural_net[layer].getDeltas(), neural_net[layer].getFilas_Deltas(), neural_net[layer].getColumnas_Deltas(), res_prom);

		// PROMEDIO DE DELTA * LEARNING_RATE
		float res_learn[1 * neural_net[layer].getColumnas_Deltas()];
		multmatfloat(res_prom, 1, neural_net[layer].getColumnas_Deltas(), res_learn, learning_rate);

		// RESULTADO NUEVAS BIAS
		neural_net[layer].setBias(resta(neural_net[layer].getBias(), 1, neural_net[layer].getNum_Neur(), res_learn));

		// actualización de pesos
		// RECUPERACIÓN DE OUTPUT CAPA ANTERIOR, Y TRANSPOSICIÓN DEL MISMO
		float output_T[output[layer_o-1].c * output[layer_o-1].f];
		transponerMatriz(output[layer_o-1].mat, output[layer_o-1].f, output[layer_o-1].c, output_T);

		// OUTPUT_T @ DELTA CAPA ACTUAL
		float res_dot[output[layer_o-1].c * neural_net[layer].getColumnas_Deltas()];
		dot(output_T, output[layer_o-1].c, output[layer_o-1].f, neural_net[layer].getDeltas(), neural_net[layer].getColumnas_Deltas(), res_dot);

		// RES_DOT * LEARNING_RATE
		float res_learn2[output[layer_o-1].c * neural_net[layer].getColumnas_Deltas()];
		multmatfloat(res_dot, output[layer_o-1].c, neural_net[layer].getColumnas_Deltas(), res_learn2, learning_rate);

		// RESULTADO NUEVOS PESOS
		neural_net[layer].setWeights(resta(neural_net[layer].getWeights(), neural_net[layer].getNum_Nant(), neural_net[layer].getNum_Neur(), res_learn2));

		layer_o--;
	}
}