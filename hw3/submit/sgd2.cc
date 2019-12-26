#include <fstream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#define DIMENSION 20
#define MAXDATASIZE 4096
#define ONEHOTCOLS 2
using namespace std;
 

typedef struct {
	double x[DIMENSION];
	int y;
} Record;

Record *trainingData = NULL;
Record *testData = NULL;
int train_N, test_N;

typedef struct {
	double w[DIMENSION];
} Weight;

Weight *weight = NULL;

void init() {
	trainingData = (Record*)malloc(sizeof(Record) * MAXDATASIZE);
	testData = (Record*)malloc(sizeof(Record) * MAXDATASIZE);
	weight = (Weight*)malloc(sizeof(Weight));
	train_N = test_N = 0;

	for (int i = 0; i < DIMENSION; ++i) {
		weight->w[i] = 0.0;
	}
}

int sign(double x){ 
	if(x > 0) return 1;
	return -1;
}
 
void getData(fstream &datafile, Record *data, int train) {
	while (!datafile.eof()) {
		Record temp;
		for(int i = 0; i < DIMENSION; ++i)
			datafile >> temp.x[i];
		datafile >> temp.y;
		if (train) {
			data[train_N++] = temp;
		}
		else {
			data[test_N++] = temp;
		}
	}
	datafile.close();
}
 
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double vectorMul(double *a, double *b, int dimension) { 
	double temp = 0.0;
	for(int i = 0; i < dimension; ++i)
		temp += a[i] * b[i];
	return temp;
}

void calcuBatchGradient(Record *data, Weight *weight, int N, double *grad) {
	for (int i = 0; i < N; ++i) {
		double temp = sigmoid(-1 * vectorMul(weight->w, data[i].x, DIMENSION) * (double)data[i].y);
		for (int j = 0; j < DIMENSION; ++j)
			grad[j] += -1.0 * temp * data[i].x[j] * data[i].y; 
	}
	for(int i = 0; i < DIMENSION; ++i)
		grad[i] = grad[i] / N;
}
 
void calcuStochasticGradient(Record data, Weight *weight, double *grad) {
	double temp = sigmoid(-1 * vectorMul(weight->w, data.x, DIMENSION) * (double)data.y);
	for (int j = 0; j < DIMENSION; ++j)
		grad[j] += -1.0 * temp * data.x[j] * data.y;
 
}
 
void updateW(Weight *weight, double eta, double *grad) {
	for (int i = 0; i < DIMENSION; ++i) {
		weight->w[i] = weight->w[i] - (eta * grad[i]);
	}
}
 
double calcuLGError(Record *data, Weight *weight, int N){
	double error = 0.0;
	for (int i = 0; i < N; i++){
		error += log(1 + exp(-data[i].y * vectorMul(weight->w, data[i].x, DIMENSION)));
	}
	return double(error / N);
}
 
void logisticRegression(Record *data, Weight *weight, int N, double eta, int iter) {
    for (int i = 0; i < iter; ++i) {
		double grad[DIMENSION] = {0.0};
		calcuBatchGradient(data, weight, N, grad);
		updateW(weight, eta, grad);
		printf("iter = %d, Logloss = %f\n", i, calcuLGError(data, weight, N));
	}
}

double calcuError(Record *data, Weight *weight, int N) {
	double error = 0.0;
	for (int i = 0; i < N; ++i) {
		if (sign(vectorMul(data[i].x, weight->w, DIMENSION)) != data[i].y)
			++error;
	}
	return (double)error / N;
}

void clean() {
	free(trainingData);
	free(testData);
	free(weight);
}

int main() {
	fstream file1("my_data/train.dat");
	fstream file2("my_data/test.dat");
	if (file1.is_open() && file2.is_open()) {
		init();
		getData(file1, trainingData, 1);
		getData(file2, testData, 0);
	}
	else {
		puts("can not open file!");
		exit(1);
	}

	double eta = 0.01;
	int iter = 2000;

	
	logisticRegression(trainingData, weight, train_N, eta, iter);
	printf("Ein = %f\n", calcuError(trainingData, weight, train_N));
    printf("Eout = %f\n\n", calcuError(testData, weight, test_N));

    clean();
}