#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#define DIMENSION 20
using namespace std;
 

struct Record {
	double x[DIMENSION];
	int y;
};
 
struct Weight {
	double w[DIMENSION];
};
 
int sign(double x){ 
	if(x > 0)return 1;
	return -1;
}
 
void getData(fstream &datafile, vector<Record> &data) {
	while (!datafile.eof()) {
		Record temp;
		for(int i = 0; i < DIMENSION; ++i)
			datafile >> temp.x[i];
		datafile >> temp.y;
		data.push_back(temp);
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
 
void calcuBatchGradient(vector<Record> &data, Weight weight, int N, double *grad) {
	for (int i = 0; i < N; ++i) {
		double temp = sigmoid(-1 * vectorMul(weight.w, data[i].x, DIMENSION) * (double)data[i].y);
		for (int j = 0; j < DIMENSION; ++j)
			grad[j] += -1.0 * temp * data[i].x[j] * data[i].y; 
	}
	for(int i = 0; i < DIMENSION; ++i)
		grad[i] = grad[i] / N;
}

void calcuStochasticGradient(Record data, Weight weight, double *grad) {
	double temp = sigmoid(-1 * vectorMul(weight.w, data.x, DIMENSION) * (double)data.y);
	for (int j = 0; j < DIMENSION; ++j)
		grad[j] += -1.0 * temp * data.x[j] * data.y;
 
}
 
void updateW(Weight &weight, double eta, double *grad) {
	for (int i = 0; i < DIMENSION; ++i) {
		weight.w[i] = weight.w[i] - (eta * grad[i]);
	}
}
 
double calcuLGError(vector<Record> &data, Weight weight, int N) {
	double error = 0.0;
	for (int i = 0; i < N; i++){
		error += log(1 + exp(-data[i].y * vectorMul(weight.w, data[i].x, DIMENSION)));
	}
	return double(error / N);
}
 
void logisticRegression(vector<Record> &data, Weight &weight, int N, double eta, int iter) {
    for (int i = 0; i < iter; ++i) {
		double grad[DIMENSION] = {0.0};
		calcuBatchGradient(data, weight, N, grad);
		updateW(weight, eta, grad);
		cout << "iter = " << i << ", Logloss = " << calcuLGError(data, weight, N) << endl;
	}
}
 
double calcuError(vector<Record> &data, Weight weight, int N) {
	double error = 0.0;
	for (int i = 0; i < N; ++i) {
		if (sign(vectorMul(data[i].x, weight.w, DIMENSION)) != data[i].y)
			++error;
	}
	return double(error / N);
}
 
int main() {
	vector<Record> trainingData;
	vector<Record> testData;
	fstream file1("my_data/train.dat");
	fstream file2("my_data/test.dat");

	if (file1.is_open() && file2.is_open()) {
		getData(file1, trainingData);
		getData(file2, testData);
	}
	else {
		cout << "can not open file!" << endl;
		exit(1);
	}
	int train_N = trainingData.size();
	int test_N = testData.size();
	double eta = 0.01;
	int iter = 2000;
	Weight weight;
	for (int i = 0; i < DIMENSION; ++i)
		weight.w[i] = 0;
	logisticRegression(trainingData, weight, train_N, eta, iter);
	cout<< "Ein = " << calcuError(trainingData, weight, train_N) << endl;
    cout<< "Eout = " << calcuError(testData, weight, test_N) << endl;
}