#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

void mostrarMatriz(float ar[], int filas, int columnas);
void transponerMatriz(float ar[], int filas, int columnas, float res[]);
void dot(float ara[], int filas_a, int columnas_a, float arb[], int columnas_b, float arr[]);
void sigmoidal(float ar[], int filas_a, int columnas_a, float arr[]);
void derivada_sigmoidal(float ar[], int filas, int columnas, float arr[]);
void resta(float aa[], int filas, int columnas, float ab[], float arr[]);
void suma(float aa[], int filas, int columnas, float ab[], float arr[]);
void multiplicar(float aa[], int filas, int columnas, float ab[], float arr[]);

class NeuralNet{
	private:
		float weightsN1[3*1];

	public:
		NeuralNet();
		void getWeights();
		void train(float set_input[], int filas_i, int columnas_i, float set_output_T[], int filas_oT, int columnas_oT, int iteraciones);
		void think(float set_input[], int filas_i, int columnas_i, float output[]);
};

NeuralNet::NeuralNet(){
	srand(time(NULL));
	weightsN1[0]= -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
	weightsN1[1]= -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
	weightsN1[2]= -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
}

void NeuralNet::getWeights(){
	mostrarMatriz(weightsN1,3,1);
}

void NeuralNet::think(float set_input[], int filas_i, int columnas_i, float output[]){
	// Si set_input 4x3 y weightsN1 3x1, entonces: res_mult 4x1
	float res_mult[filas_i*1];
	dot(set_input,filas_i,columnas_i,weightsN1,1,res_mult);

	// Si res_mult 4x1, entonces: output 4x1
	sigmoidal(res_mult,filas_i,1,output);
}

void NeuralNet::train(float set_input[], int filas_i, int columnas_i, float set_output_T[], int filas_oT, int columnas_oT, int iteraciones){
	float output[4*1];
	float error[4*1];
	float set_input_T[columnas_i*filas_i];
	float der_sigm[4*1];
	float error_x_dersigm[4*1];
	float adjustment[3*1];

	for(int i=0;i<iteraciones;i++){
		think(set_input,filas_i,columnas_i,output); // Result: output 4x1
		resta(set_output_T,filas_oT,columnas_oT,output,error); // Result: error 4x1
		
		transponerMatriz(set_input,filas_i,columnas_i,set_input_T); // Result: set_input_T 3x4
		derivada_sigmoidal(output,4,1,der_sigm); // Result: der_sigm 4x1
		multiplicar(error,4,1,der_sigm,error_x_dersigm); // Result: error_x_dersigm 4*1

		dot(set_input_T,columnas_i,filas_i,error_x_dersigm,1,adjustment);

		suma(adjustment,3,1,weightsN1,weightsN1);
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

void transponerMatriz(float ar[], int filas, int columnas, float res[]){
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			res[(j*filas)+i]=ar[(i*columnas)+j];
		}
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

void resta(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(unsigned i=0;i<filas;i++){
		for(unsigned j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]-ab[(i*columnas)+j];
		}
	}
}

void suma(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(unsigned i=0;i<filas;i++){
		for(unsigned j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]+ab[(i*columnas)+j];
		}
	}
}

void multiplicar(float aa[], int filas, int columnas, float ab[], float arr[]){
	for(unsigned i=0;i<filas;i++){
		for(unsigned j=0;j<columnas;j++){
			arr[(i*columnas)+j]=aa[(i*columnas)+j]*ab[(i*columnas)+j];
		}
	}
}

int main(){
	NeuralNet n;

	cout<<"\n > Random starting synaptic weights:\n";
	n.getWeights();

	float set_input[4*3]={0,0,1 ,1,1,1, 1,0,1, 0,1,1};

	float set_output[1*4]={0,1,1,0};
	float set_output_T[4*1];

	float res[1*1];

	transponerMatriz(set_output, 1, 4, set_output_T);
	//mostrarMatriz(mat_output_T,4,1);

	n.train(set_input, 4, 3, set_output_T, 4, 1, 20000);

	cout<<"\n > New synaptic weights after training:\n";
	n.getWeights();

	float p[1*3]={1,0,0};
	n.think(p,1,3,res);
	cout<<"\n > Considering new situation [1,0,0] -> ? \n";
	mostrarMatriz(res,1,1);

	float p2[1*3]={0,0,1};
	n.think(p2,1,3,res);
	cout<<"\n > Considering [0,0,1] -> 0 \n";
	mostrarMatriz(res,1,1);

	float p3[1*3]={1,1,1};
	n.think(p3,1,3,res);
	cout<<"\n > Considering [1,1,1] -> 1 \n";
	mostrarMatriz(res,1,1);

}