#include "method.h"
#include <cstring>
#include <iostream>
#include <atomic>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

using namespace thundercat;
using namespace std;

/**
 * This solver is adapted from 
 *   Liu W., Li A., Hogg J., Duff I.S., Vinter B. (2016)
 *   A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves.
 *   In Euro-Par 2016: Parallel Processing.
 */

void EuroPar16Solver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  cscMatrix = csc;
  rowLengths = new int[cscMatrix->N];
  int *rowIndices = cscMatrix->rowIndices;
  
#pragma omp parallel for
  for (int i = 0; i < cscMatrix->NZ; i++) {
#pragma omp atomic update
    rowLengths[rowIndices[i]]++;
  }
}

void EuroPar16Solver::forwardSolve(double* __restrict b, double* __restrict x) {
  int N = cscMatrix->N;
  atomic<double> *leftsum = new atomic<double>[N];
  atomic<int> *knownVars = new atomic<int>[N];
  for (int i = 0; i < N; i++) {
    atomic_init(&(leftsum[i]), 0.0);
    atomic_init(&(knownVars[i]), 0);
  }
  int *colPtr = cscMatrix->colPtr;
  int *rowIndices = cscMatrix->rowIndices;
  double *values = cscMatrix->values;
  
#pragma omp parallel for
  for (int j = 0; j < N; j++) {
    int rowLength = rowLengths[j] - 1;
    while (rowLength != knownVars[j]) {
      // spin-wait for all the vars on this row to become known
    }
    
    double xj = (b[j] - leftsum[j]) / values[colPtr[j]];
    x[j] = xj;
    for (int k = colPtr[j] + 1; k < cscMatrix->colPtr[j+1]; k++) {
      int row = rowIndices[k];
      double mult = values[k] * xj;
      double desired, expected;
      do {
        expected = leftsum[row].load();
        desired = expected + mult;
      } while (!leftsum[row].compare_exchange_weak(expected, desired));
      knownVars[row]++;
    }
  }
  
  delete[] leftsum;
  delete[] knownVars;
}

string EuroPar16Solver::getName() {
  return "EuroPar16";
}
