#include "method.h"
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>

using namespace thundercat;
using namespace std;


ParallelCSCSolver::ParallelCSCSolver() : indexQueue() {

}

void ParallelCSCSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                             CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcscMatrix = ldcsc;
  ldcsrMatrix = ldcsr;
  udcscMatrix = udcsc;
  initialDependencies = new int[ldcscMatrix->M];

  for (int i = 0; i < ldcscMatrix->M; i++) {
    initialDependencies[i] = (ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i]) - 1;
  }

}

void ParallelCSCSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  int solved = 0;
  int *dependencies = new int[ldcscMatrix->M];
  memcpy(dependencies, initialDependencies, sizeof(int) * ldcscMatrix->M);

  for (int i = 0; i < ldcscMatrix->M; i++) {
    initialDependencies[i] = (ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i]) - 1;
    if (dependencies[i] == 0) {
#pragma omp critical(queue)
      indexQueue.push(i);
    }
  }

#pragma omp parallel
  while (solved < ldcscMatrix->M) {

    int solve_index = -1;
    for (int wait = 0; wait < 1000000; wait++) {
      if (!indexQueue.empty() || solved >= ldcscMatrix->M) break;
    }

#pragma omp critical(queue)
    {
      if (!(indexQueue.empty())) {
        solve_index = indexQueue.front();
        indexQueue.pop();
      }
    }
    if (solve_index >= 0) {
      double componentSum = 0.0;
      for (
          int componentIndex = ldcsrMatrix->rowPtr[solve_index];
          componentIndex < ldcsrMatrix->rowPtr[solve_index + 1] - 1;
          componentIndex++) {
        componentSum +=
            x[ldcsrMatrix->colIndices[componentIndex]] * ldcsrMatrix->values[componentIndex];

      }
      x[solve_index] = (b[solve_index] - componentSum) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];
        int ready;
#pragma omp atomic capture seq_cst
        ready = --(dependencies[ldcscMatrix->rowIndices[i]]);
        if (ready == 0) {
#pragma omp critical(queue)
          indexQueue.push(ldcscMatrix->rowIndices[i]);
        }
      }
#pragma omp atomic update seq_cst
      solved += 1;
    }
  }

  delete[] dependencies;
}

void ParallelCSCSolver::backwardSolve(double *__restrict b, double *__restrict x) {
  double *rightsum = new double[udcscMatrix->M];
  memset(rightsum, 0, sizeof(double) * udcscMatrix->M);
  for (int j = udcscMatrix->M - 1; j >= 0; j--) {
    x[j] = (b[j] - rightsum[j]) / udcscMatrix->values[udcscMatrix->colPtr[j + 1] - 1];
//#pragma omp parallel for
    for (int k = udcscMatrix->colPtr[j + 1] - 1 - 1; k >= udcscMatrix->colPtr[j]; k--) {
      int row = udcscMatrix->rowIndices[k];
      rightsum[row] += udcscMatrix->values[k] * x[j];
    }
  }

  delete[] rightsum;
}

string ParallelCSCSolver::getName() {
  return "ParallelCSC";
}
