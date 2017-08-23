#include "method.h"
#include <cstring>
#include <iostream>

using namespace thundercat;
using namespace std;

void SequentialCSRSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                               CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcsrMatrix = ldcsr;
  udcsrMatrix = udcsr;
}

void SequentialCSCSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                               CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcscMatrix = ldcsc;
  udcscMatrix = udcsc;
}

void SequentialCSRSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  for (int i = 0; i < ldcsrMatrix->N; i++) {
    double sum = 0.0;
    int k;
    for (k = ldcsrMatrix->rowPtr[i]; k < ldcsrMatrix->rowPtr[i+1] - 1; k++) {
      int col = ldcsrMatrix->colIndices[k];
      sum += ldcsrMatrix->values[k] * x[col];
    }
    x[i] = (b[i] - sum) / ldcsrMatrix->values[k];
  }
}

void SequentialCSRSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  for (int i = udcsrMatrix->N - 1; i >= 0; i--) {
    double sum = 0.0;
    int k;
    for (k = udcsrMatrix->rowPtr[i+1] - 1; k > udcsrMatrix->rowPtr[i]; k--) {
      int col = udcsrMatrix->colIndices[k];
      sum += udcsrMatrix->values[k] * x[col];
    }
    x[i] = (b[i] - sum) / udcsrMatrix->values[k];
  }
}

void SequentialCSCSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  double *leftsum = new double[ldcscMatrix->M];
  memset(leftsum, 0, sizeof(double) * ldcscMatrix->M);
  
  for (int j = 0; j < ldcscMatrix->M; j++) {
    x[j] = (b[j] - leftsum[j]) / ldcscMatrix->values[ldcscMatrix->colPtr[j]];
    for (int k = ldcscMatrix->colPtr[j] + 1; k < ldcscMatrix->colPtr[j+1]; k++) {
      int row = ldcscMatrix->rowIndices[k];
      leftsum[row] += ldcscMatrix->values[k] * x[j];
    }
  }
  
  delete[] leftsum;
}

void SequentialCSCSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  double *rightsum = new double[udcscMatrix->M];
  memset(rightsum, 0, sizeof(double) * udcscMatrix->M);
  
  for (int j = udcscMatrix->M - 1; j >= 0; j--) {
    x[j] = (b[j] - rightsum[j]) / udcscMatrix->values[udcscMatrix->colPtr[j+1]-1];
    for (int k = udcscMatrix->colPtr[j+1] - 1 - 1; k >= udcscMatrix->colPtr[j]; k--) {
      int row = udcscMatrix->rowIndices[k];
      rightsum[row] += udcscMatrix->values[k] * x[j];
    }
  }
  
  delete[] rightsum;
}

string SequentialCSRSolver::getName() {
  return "SequentialCSR";
}

string SequentialCSCSolver::getName() {
  return "SequentialCSC";
}
