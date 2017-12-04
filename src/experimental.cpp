#include "method.h"
#include <cstring>
#include <iostream>
#include <atomic>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

using namespace thundercat;
using namespace std;

void ExperimentalSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                              CSRMatrix *udcsr, CSCMatrix *udcsc, int iters) {
  ldcsrMatrix = ldcsr;
  ldcscMatrix = ldcsc;
  udcsrMatrix = udcsr;
  udcscMatrix = udcsc;
  unknownVars = new int[ldcscMatrix->N];
  rowsToSolve.resize(omp_get_max_threads());
  
  #pragma omp parallel for
  for (int i = 0; i < ldcsrMatrix->N; i++) {
    int threadId = omp_get_thread_num();
    int length = ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i];
    unknownVars[i] = length - 1;
    if (unknownVars[i] == 0) {
      rowsToSolve[threadId].push_back(i);
    }
  }
}

void ExperimentalSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  int N = ldcscMatrix->N;
  atomic<int> *knownVars = new atomic<int>[N];
  std::vector<std::deque<int> > assignedRows;
  assignedRows.resize(omp_get_max_threads());

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    knownVars[i] = 0;
  }
  
  #pragma omp parallel
  {
    int threadId = omp_get_thread_num();
    assignedRows[threadId] = rowsToSolve[threadId];
    while (!assignedRows[threadId].empty()) {
      int i = assignedRows[threadId].front();
      //printf("%d ", i);
      assignedRows[threadId].pop_front();
      double leftsum = 0;
      int j;
      for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
	int col = ldcsrMatrix->colIndices[j];
	leftsum += ldcsrMatrix->values[j] * x[col];
      }
      double xi = (b[i] - leftsum) / ldcsrMatrix->values[j];
      x[i] = xi;
      for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
	int row = ldcscMatrix->rowIndices[k];
	if (++(knownVars[row]) == unknownVars[row]) {
	  assignedRows[threadId].push_back(row);
	}
      }
    }
  }
  delete[] knownVars;
}

void ExperimentalSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  // No parallelization yet.
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

string ExperimentalSolver::getName() {
  return "Experimental";
}
