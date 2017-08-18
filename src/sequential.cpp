#include "method.h"
#include <cstring>

using namespace thundercat;
using namespace std;

void SequentialCSRSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  csrMatrix = csr;
}

void SequentialCSCSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  cscMatrix = csc;
}

void SequentialCSRSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  for (int i = 0; i < csrMatrix->N; i++) {
    double sum = 0.0;
    int k;
    for (k = csrMatrix->rowPtr[i]; k < csrMatrix->rowPtr[i+1] - 1; k++) {
      int col = csrMatrix->colIndices[k];
      sum += csrMatrix->values[k] * x[col];
    }
    x[i] = (b[i] - sum) / csrMatrix->values[k];
  }
}

void SequentialCSCSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  double *leftsum = new double[cscMatrix->M];
  memset(leftsum, 0, sizeof(double) * cscMatrix->M);
  
  for (int j = 0; j < cscMatrix->M; j++) {
    x[j] = (b[j] - leftsum[j]) / cscMatrix->values[cscMatrix->colPtr[j]];
    for (int k = cscMatrix->colPtr[j] + 1; k < cscMatrix->colPtr[j+1]; k++) {
      int row = cscMatrix->rowIndices[k];
      leftsum[row] += cscMatrix->values[k] * x[j];
    }
  }
  
  delete[] leftsum;
}

string SequentialCSRSolver::getName() {
  return "SequentialCSR";
}

string SequentialCSCSolver::getName() {
  return "SequentialCSC";
}
