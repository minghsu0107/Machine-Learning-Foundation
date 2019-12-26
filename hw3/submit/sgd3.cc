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
	double w[ONEHOTCOLS][DIMENSION];
} Weight;

Weight *weight = NULL;

double probs[MAXDATASIZE][ONEHOTCOLS];
double probs_trans[ONEHOTCOLS][MAXDATASIZE];
double trainingData_trans[DIMENSION][MAXDATASIZE];
int y_train_onehot[MAXDATASIZE][ONEHOTCOLS] = {};

void show_weight() {
	for (int i = 0; i < ONEHOTCOLS; ++i) {
		printf("col %d\n", i);
    	for (int j = 0; j < DIMENSION; ++j) {
    		printf("%lf ", weight->w[i][j]);
    	}
    	puts("");
    }
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

void init() {
	trainingData = (Record*)malloc(sizeof(Record) * MAXDATASIZE);
	testData = (Record*)malloc(sizeof(Record) * MAXDATASIZE);
	weight = (Weight*)malloc(sizeof(Weight));
	train_N = test_N = 0;

	for (int i = 0; i < ONEHOTCOLS; ++i) {
		for (int j = 0; j < DIMENSION; ++j) {
			weight->w[i][j] = 0.0;
		}
	}
}

void OneHotEncoding() {
	for (int i = 0; i < train_N; ++i)
		y_train_onehot[i][trainingData[i].y] = 1;
}

void prepocess() {
	for (int i = 0; i < train_N; ++i)
    	trainingData[i].y = trainingData[i].y == -1? 0: 1;
    for (int i = 0; i < test_N; ++i)
    	testData[i].y = testData[i].y == -1? 0: 1;
	OneHotEncoding();
}

void softmax(double *scores, int N) {
	int i;
    double sum, max;

    for (i = 1, max = scores[0]; i < N; ++i) {
        if (scores[i] > max) {
            max = scores[i];
        }
    }

    for (i = 0, sum = 0; i < N; ++i) {
        scores[i] = exp(scores[i] - max);
        sum += scores[i];
    }

    for (i = 0; i < N; ++i) {
        scores[i] /= sum;
    }
}

double vectorMul(double *a, double *b, int dimension) { 
	double temp = 0.0;
	for(int i = 0; i < dimension; ++i)
		temp += a[i] * b[i];
	return temp;
}

void get_prob_transpos() {
	for (int i = 0; i < ONEHOTCOLS; ++i) {
        for (int j = 0; j < MAXDATASIZE; ++j) {
            probs_trans[i][j] = probs[j][i];
        }
	}
}

void calcuBatchGradient(Record *data, Weight *weight, int N, double grad[ONEHOTCOLS][DIMENSION]) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < ONEHOTCOLS; ++j) {
			probs[i][j] = vectorMul(weight->w[j], data[i].x, DIMENSION);
		}
		softmax(probs[i], ONEHOTCOLS);
		for (int j = 0; j < ONEHOTCOLS; ++j) {
			probs[i][j] -= y_train_onehot[i][j];
		}
	}
	get_prob_transpos();
	for (int i = 0; i < ONEHOTCOLS; ++i) {
		for (int j = 0; j < DIMENSION; ++j) {
			grad[i][j] = vectorMul(trainingData_trans[j], probs_trans[i], N);
		}
	}
}

void updateW(Weight *weight, double eta, double grad[ONEHOTCOLS][DIMENSION]) {
	for (int i = 0; i < ONEHOTCOLS; ++i) {
		for (int j = 0; j < DIMENSION; ++j) {
			weight->w[i][j] = weight->w[i][j] - (eta * grad[i][j]);
		}
	}
}

void multiclass_classifier(Record *data, Weight *weight, int N, double eta, int iter) {
    for (int i = 0; i < iter; ++i) {
		double grad[ONEHOTCOLS][DIMENSION];
		calcuBatchGradient(data, weight, N, grad);
		updateW(weight, eta, grad);
	}
}

double calcuError(Record *data, Weight *weight, int N) {
	double error = 0.0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < ONEHOTCOLS; ++j) {
			probs[i][j] = vectorMul(data[i].x, weight->w[j], DIMENSION);
		}
		softmax(probs[i], ONEHOTCOLS);

		int idx = 0;
		double M = probs[i][0];
		for (int j = 1; j < ONEHOTCOLS; ++j) {
        	if (probs[i][j] > M) {
            	M = probs[i][j];
            	idx = j;
        	}
    	}
    	if (idx != data[i].y)
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
		for (int i = 0; i < DIMENSION; ++i) {
	        for (int j = 0; j < train_N; ++j) {
	            trainingData_trans[i][j] = trainingData[j].x[i];
	        }
		}
	}
	else {
		puts("can not open file!");
		exit(1);
	}

	double eta = 0.01;
	int iter = 2000;

	prepocess();
	multiclass_classifier(trainingData, weight, train_N, eta, iter);
	printf("Ein = %lf\n", calcuError(trainingData, weight, train_N));
	printf("Eout = %lf\n\n", calcuError(testData, weight, test_N));

	show_weight();
	clean();
}