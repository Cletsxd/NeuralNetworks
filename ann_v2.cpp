#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

// DECLARACIÓN DE LA CLASE NeuralLayer, que contendrá los atributos necesarios para una capa de la red neuronal
class NeuralLayer{
	private:
		// Pesos de las sinapsis de la capa
		float* weights;

		// Bias de las neuronas
		float* bias;

		// Deltas
		float* deltas;
		int filas_d;
		int columnas_d;

		// Número de conexiones sinápticas (filas) antes de llegar a la capa
		int num_con;

		// Número de neuronas (columnas) de la capa
		int num_neur;

		// Asigna valores a los Pesos (weights)
		void setWeights();

		// Asigna valores a las Bias (bias)
		void setBias();

		// Igualar deltas
		void igualarDeltas(float* aa, int filas, int columnas, float* ab);

	public:
		// Crea una capa vacía
		NeuralLayer();

		// Asigna  valores a la capa de la red
		// Parámetros: número de conexiones (num_con) y neuronas (num_neur) de la capa
		void setValuesNeuralLayer(int num_con, int num_neur);

		// Asigna valores a las deltas de la capa
		void setDeltas(float deltas[], int filas_d, int columnas_d);

		// Regresa los datos de la delta
		float* getDeltas();
		int getFilas_D();
		int getColumnas_D();

		// Regresa el apuntador de los pesos de la capa
		float* getWeights();

		// Regresa el apuntador de las bias de la capa
		float* getBias();

		// Regresa la cantidad de conexiones sináptinas de la capa
		int getNum_Con();

		// Regresa la cantidad de neuronas de la capa
		int getNum_Neur();
};

// IMPLEMENTACIÓN DE LAS FUNCIONES DE LA CLASE

NeuralLayer::NeuralLayer(){

}

void NeuralLayer::setWeights(){
	this->weights = new float[this->num_con * this->num_neur];

	srand(time(NULL));
	for(int i=0;i<this->num_con;i++){
		for(int j=0;j<this->num_neur;j++){
			this->weights[(i*this->num_neur)+j] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
		}
	}
}

void NeuralLayer::setBias(){
	this->bias = new float[1 * this->num_neur];

	srand(time(NULL));
	for(int i=0;i<1;i++){
		for(int j=0;j<this->num_neur;j++){
			this->bias[(i*this->num_neur)+j] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
		}
	}
}

float* NeuralLayer::getWeights(){
	return this->weights;
}

float* NeuralLayer::getBias(){
	return this->bias;
}

int NeuralLayer::getNum_Con(){
	return this->num_con;
}

int NeuralLayer::getNum_Neur(){
	return this->num_neur;
}

void NeuralLayer::setValuesNeuralLayer(int num_con, int num_neur){
	this->num_con = num_con;
	this->num_neur = num_neur;
	setWeights();
	setBias();
}

void NeuralLayer::igualarDeltas(float* aa, int filas, int columnas, float* ab){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			ab[(i*columnas)+j] = aa[(i*columnas)+j];
		}
	}
}

void NeuralLayer::setDeltas(float deltas[], int filas_d, int columnas_d){
	this->deltas = new float[filas_d * columnas_d];
	igualarDeltas(deltas, filas_d, columnas_d, this->deltas);
	this->filas_d = filas_d;
	this->columnas_d = columnas_d;
}

float* NeuralLayer::getDeltas(){
	return this->deltas;
}

int NeuralLayer::getFilas_D(){
	return this->filas_d;
}

int NeuralLayer::getColumnas_D(){
	return this->columnas_d;
}

// DECLARACIÓN ESPECIAL DE LAS SALIDAS DE LAS CAPAS
struct Output{
		float* mat;
		int f;
		int c;
};

// DECLARACIÓN DE LAS FUNCIONES GLOBALES

// Crea la red neuronal
// Parámetros: topología (nn_topology[]), cantidad de capas después de la capa de entrada (n_layers)
// y arreglo de capas referente a la red neuronal (neural_net[])
void createNeuralNet(float nn_topology[], int n_layers, NeuralLayer neural_net[]);

// Mostrar matriz
// Parámetros: la matriz (ar[]), filas y columnas
void mostrarMatriz(float ar[], int filas, int columnas);

// Multiplicación matricial
void dot(float ara[], int filas_a, int columnas_a, float arb[], int columnas_b, float arr[]);

// Suma de las BIAS a la matriz ponderada
void sumaBias(float mat_ponderada[], int filas_mp, int columnas_mp, float mat_bias[], float res[]);

// Función de activación: TANGENTE HIPERBÓLICA
void tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]);

// Derivada de la función TANGENTE HIPERBÓLICA
void derivada_tang_hiper(float ar[], int filas_a, int columnas_a, float arr[]);

// Función de activación: SIGMOIDAL
void sigmoidal(float ar[], int filas_a, int columnas_a, float arr[]);

// Derivada de la función SIGMOIDAL
void derivada_sigmoidal(float ar[], int filas, int columnas, float arr[]);

// Función de activación
void activationFunction(float weigthed_mat[], int filas, int columnas, float res[]);

// Derivada de la función de activación
void derivActivationFunction(float activated_mat[], int filas, int columnas, float res[]);

// Derivada del ERROR CUADRÁTICO MEDIO
void derivada_e2medio(float aa[], int filas, int columnas, float ab[], float arr[]);

// Multiplicación no matricial de matrices
void multiplicar(float aa[], int filas, int columnas, float ab[], float arr[]);

// Transpone una matriz
void transponerMatriz(float ar[], int filas, int columnas, float res[]);

// Multiplica una matriz por un número flotante
void multmatfloat(float aa[], int filas, int columnas, float res[], float learning_rate);

// Resta dos matrices
void resta(float aa[], int filas, int columnas, float ab[], float arr[]);

// Promedio de matriz por cada columnas
void mean(float aa[], int filas, int columnas, float arr[]);

// Inicia una matriz con ceros
void zeros(float a[], int filas, int columnas);

// Pensar, muestra un resultado a partir de una entrada. Prueba 1
void think1(float entrada[], int filas_entrada, int columnas_entrada, float res[], NeuralLayer* neural_net, int n_layers);

// Pensar, recupera los valores de las salidas por capas. Prueba 2
void think2(float entrada[], int filas_entrada, int columnas_entrada, Output output[], NeuralLayer* neural_net, int n_layers);

// Actualiza los pesos de una capa de la red
void weightsActualization(NeuralLayer neural_layer, Output output, float learning_rate);

// Algoritmo de retropropagación
void backpropagation(float salida[], float filas_salida, float columnas_salida, Output output[], NeuralLayer neural_net[], int n_layers, float learning_rate);

// Entrenamiento. Prueba 1
void train1(float entrada[], int filas_entrada, int columnas_entrada, float salida[], float filas_salida, float columnas_salida, NeuralLayer neural_net[], int n_layers, float learning_rate);

// IMPLEMENTACIÓN DE LAS FUNCIONES GLOBALES

void createNeuralNet(float nn_topology[], int n_layers, NeuralLayer neural_net[]){
	// Mientras recorre la topología, crea la red neuronal
	for(int layer = 0; layer<n_layers;layer++){
		neural_net[layer].setValuesNeuralLayer(nn_topology[layer], nn_topology[layer+1]);
	}
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

void multiplicar(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]*ab[(i*columnas)+j];
		}
	}
}

void transponerMatriz(float ar[], int filas, int columnas, float res[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			res[(j*filas)+i]=ar[(i*columnas)+j];
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

void resta(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]-ab[(i*columnas)+j];
		}
	}
}

void mean(float aa[], int filas, int columnas, float arr[]){
	// Sumar matriz por columnas
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			arr[j]=(aa[(i*columnas)+j]/2)+arr[j];
		}
	}
}

void zeros(float a[], int filas, int columnas){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			a[(i*columnas)+j]=0;
		}
	}
}

void think1(float entrada[], int filas_entrada, int columnas_entrada, float res[], NeuralLayer* neural_net, int n_layers){
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
	cout<<"neural_net[0].getWeights()"<<endl;
	mostrarMatriz(neural_net[0].getWeights(), neural_net[0].getNum_Con(), neural_net[0].getNum_Neur());
}

void think2(float entrada[], int filas_entrada, int columnas_entrada, Output output[], NeuralLayer* neural_net, int n_layers){
	cout<<"think(){"<<endl;
	cout<<"declaracion de output"<<endl;
	output[0].mat = entrada;
	output[0].f = filas_entrada;
	output[0].c = columnas_entrada;
	cout<<"fin declaracion de output"<<endl;

	cout<<"i_for"<<endl;
	for(int layer_o=1;layer_o<n_layers;layer_o++){
		cout<<"declaracion res_temp"<<endl;
		cout<<"layer_o = "<<layer_o<<endl;
		cout<<"layer_o-1 = "<<layer_o-1<<endl;
		cout<<"output[layer_o-1].f = "<<output[layer_o-1].f<<endl;
		// LA LÍNEA 401, DA UN ERROR LÓGICO, QUE PUEDE SER DE MEMORIA, CUANDO EL VALOR DE LA DIRECCIÓN NO TIENE UN VALOR YA ASIGNADO
		cout<<"neural_net[layer_o-1].getNum_Neur() = "<<neural_net[layer_o-1].getNum_Neur()<<endl;
		float res_temp[output[layer_o-1].f * neural_net[layer_o-1].getNum_Neur()];
		cout<<"f_declaracion res_temp"<<endl;
		cout<<"i_dot"<<endl;
		dot(output[layer_o-1].mat, output[layer_o-1].f, output[layer_o-1].c, neural_net[layer_o-1].getWeights(), neural_net[layer_o-1].getNum_Neur(), res_temp);
		cout<<"f_dot"<<endl;
		// Variable 'columnas' cambiará después de la multiplicación matricial
		output[layer_o].c = neural_net[layer_o-1].getNum_Neur();

		cout<<"i_sumaBias"<<endl;
		sumaBias(res_temp, output[layer_o-1].f, output[layer_o].c, neural_net[layer_o-1].getBias(), res_temp);
		cout<<"f_sumaBias"<<endl;

		output[layer_o].mat = new float[output[layer_o-1].f * output[layer_o].c];
		cout<<"i_activationFunc"<<endl;
		activationFunction(res_temp, output[layer_o-1].f, output[layer_o].c, output[layer_o].mat);
		cout<<"f_activationFunc"<<endl;
		output[layer_o].f=output[layer_o-1].f;
	}
	cout<<"fi_for"<<endl;
	cout<<"think()}"<<endl;
}

void weightsActualization(NeuralLayer neural_layer, Output output, float learning_rate){
	float* deltas_temp = neural_layer.getDeltas();
	int filas_d = neural_layer.getFilas_D();
	int columnas_d = neural_layer.getColumnas_D();

	float output_T[output.c * output.f];
	int filas_oT = output.c;
	int columnas_oT = output.f;
	transponerMatriz(output.mat, output.f, output.c, output_T);

	float res_temp1[filas_oT * columnas_d];
	dot(output_T, filas_oT, columnas_oT, deltas_temp, columnas_d, res_temp1);

	float res_temp2[filas_oT * columnas_d];
	multmatfloat(res_temp1, filas_oT, columnas_d, res_temp2, learning_rate);

	resta(neural_layer.getWeights(), neural_layer.getNum_Con(), neural_layer.getNum_Neur(), res_temp2, neural_layer.getWeights());
}

void biasActualization(NeuralLayer neural_layer, float learning_rate){
	float* deltas_temp = neural_layer.getDeltas();
	int filas_d = neural_layer.getFilas_D();
	int columnas_d = neural_layer.getColumnas_D();

	float res_temp1[1*columnas_d];
	zeros(res_temp1, 1, columnas_d);
	mean(deltas_temp, filas_d, columnas_d, res_temp1);

	float res_temp2[1*columnas_d];
	multmatfloat(res_temp1, 1, columnas_d, res_temp2, learning_rate);

	resta(neural_layer.getBias(), 1, neural_layer.getNum_Neur(), res_temp2, neural_layer.getBias());
}

void backpropagation(float salida[], float filas_salida, float columnas_salida, Output output[], NeuralLayer neural_net[], int n_layers, float learning_rate){
	for(int layer=n_layers-1;layer>=0;layer--){
		if(layer==n_layers-1){
			// última capa
			int layer_o = layer+1;

			float res_temp1[output[layer_o].f * output[layer_o].c];
			derivada_e2medio(output[layer_o].mat, output[layer_o].f, output[layer_o].c, salida, res_temp1);
			//derivada_e2medio(salida, filas_salida, columnas_salida, output[layer_o].mat, res_temp1);

			float res_temp2[output[layer_o].f * output[layer_o].c];
			derivActivationFunction(output[layer_o].mat, output[layer_o].f, output[layer_o].c, res_temp2);

			float res_temp3[output[layer_o].f * output[layer_o].c];
			multiplicar(res_temp1, output[layer_o].f, output[layer_o].c, res_temp2, res_temp3);
			neural_net[layer].setDeltas(res_temp3, output[layer_o].f, output[layer_o].c);
		}else{
			// capas anteriores
			int layer_o=layer+1;

			float* weights_temp = neural_net[layer].getWeights();
			int filas_w = neural_net[layer].getNum_Con();
			int columnas_w = neural_net[layer].getNum_Neur();

			float weights_T[columnas_w * filas_w];
			int filas_wT = columnas_w;
			int columnas_wT = filas_w;
			transponerMatriz(weights_temp, filas_w, columnas_w, weights_T);

			float res_temp1[neural_net[layer+1].getFilas_D() * columnas_wT];
			dot(neural_net[layer+1].getDeltas(), neural_net[layer+1].getFilas_D(), neural_net[layer+1].getColumnas_D(), weights_T, columnas_wT, res_temp1);

			float res_temp2[output[layer_o].f * output[layer_o].c];
			derivActivationFunction(output[layer_o].mat, output[layer_o].f, output[layer_o].c, res_temp2);

			float res_temp3[neural_net[layer+1].getFilas_D() * columnas_wT];
			multiplicar(res_temp1, neural_net[layer+1].getFilas_D(), columnas_wT, res_temp2, res_temp3);

			neural_net[layer].setDeltas(res_temp3, neural_net[layer+1].getFilas_D(), columnas_wT);
		}

		// actualización de pesos y bias
		weightsActualization(neural_net[layer], output[layer], learning_rate);
		biasActualization(neural_net[layer], learning_rate);
	}
}

void train1(float entrada[], int filas_entrada, int columnas_entrada, float salida[], float filas_salida, float columnas_salida, NeuralLayer neural_net[], int n_layers, float learning_rate){
	Output* output = new Output[n_layers+1];

	cout<<"i_think2"<<endl;
	cout<<"neural_net[0].getWeights()"<<endl;
	mostrarMatriz(neural_net[0].getWeights(), neural_net[0].getNum_Con(), neural_net[0].getNum_Neur());
	think2(entrada, filas_entrada, columnas_entrada, output, neural_net, n_layers+1);
	cout<<"f_think2"<<endl;

	// Retropropagación con descenso del gradiente
	cout<<"i_backpr"<<endl;
	backpropagation(salida, filas_salida, columnas_salida, output, neural_net, n_layers, learning_rate);
	cout<<"f_backpr"<<endl;
}

int main(){
	// Capa de entrada
	float X[4*2] = {0,0, 0,1, 1,0, 1,1};

	// Resultados esperados de salida
	float Y[1*4] = {0, 1, 1, 0};

	// Número de capas
	int n_layers = 3;

	// Topología de la red neuronal
	float nn_topology[n_layers] = {2, 3, 1};

	// Se crea un arreglo de capas de neuronas. La red neuronal (neural_net[]), tendrá el valor de capas de la variable n_layers-1
	// La red neuronal tendrá solamente las capas ocultas y de salida
	NeuralLayer neural_net[n_layers-1];

	// Crear la red neuronal
	createNeuralNet(nn_topology, n_layers-1, neural_net);

	cout<<"\n > Random starting synaptic weights:\n\n";
	for(int i=0;i<n_layers-1;i++){
		cout<<"\t>> Layer "<<i+1<<endl<<endl;
		
		cout<<"  Weights"<<endl;
		mostrarMatriz(neural_net[i].getWeights(), neural_net[i].getNum_Con(), neural_net[i].getNum_Neur());

		cout<<"\n  Bias"<<endl;
		mostrarMatriz(neural_net[i].getBias(), 1, neural_net[i].getNum_Neur());
		cout<<endl;
	}

	// Procedimiento de PENSAR (think1)
	float res[4*1];
	think1(X, 4, 2, res, neural_net, n_layers-1);

	cout<<" > Resultante prueba think1: "<<endl;
	mostrarMatriz(res, 4, 1);

	cout<<"neural_net[0].getWeights()"<<endl;
	mostrarMatriz(neural_net[0].getWeights(), neural_net[0].getNum_Con(), neural_net[0].getNum_Neur());

	// Procedimiento de ENTRENAR (train1)
	for(int i =0;i<5;i++){
		cout<<"\n > TRAINING "<<i+1<<endl;
		train1(X, 4, 2, Y, 1, 4, neural_net, n_layers-1, 0.5);
	}

	cout<<"\n > Synaptic weights after training:\n\n";
	for(int i=0;i<n_layers-1;i++){
		cout<<"\t>> Layer "<<i+1<<endl<<endl;
		
		cout<<"  Weights"<<endl;
		mostrarMatriz(neural_net[i].getWeights(), neural_net[i].getNum_Con(), neural_net[i].getNum_Neur());

		cout<<"\n  Bias"<<endl;
		mostrarMatriz(neural_net[i].getBias(), 1, neural_net[i].getNum_Neur());
		cout<<endl;
	}

	// Procedimiento de PENSAR (think1)
	float res1[4*1];
	think1(X, 4, 2, res1, neural_net, n_layers-1);

	cout<<" > Resultante prueba think1:"<<endl;
	mostrarMatriz(res1, 4, 1);

	// Procedimiento de PENSAR (think1)
	float X1[1*2] = {1,0};
	float res2[1*1];
	think1(X1, 1, 2, res2, neural_net, n_layers-1);
	cout<<"\n > Resultante prueba think1 {1,0} = 1: "<<endl;
	mostrarMatriz(res2, 1, 1);

	// Procedimiento de PENSAR (think1)
	float X2[1*2] = {0,1};
	float res3[1*1];
	think1(X2, 1, 2, res3, neural_net, n_layers-1);
	cout<<"\n > Resultante prueba think1 {0,1} = 1: "<<endl;
	mostrarMatriz(res3, 1, 1);
}