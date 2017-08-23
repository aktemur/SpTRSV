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

void EuroPar16Solver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                           CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcscMatrix = ldcsc;
  udcscMatrix = udcsc;
  ldrowLengths = new int[ldcscMatrix->N];
  udrowLengths = new int[udcscMatrix->N];
  
#pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->NZ; i++) {
#pragma omp atomic update
    ldrowLengths[ldcscMatrix->rowIndices[i]]++;
  }
#pragma omp parallel for
  for (int i = 0; i < udcscMatrix->NZ; i++) {
#pragma omp atomic update
    udrowLengths[udcscMatrix->rowIndices[i]]++;
  }
}

void EuroPar16Solver::forwardSolve(double* __restrict b, double* __restrict x) {
  int N = ldcscMatrix->N;
  atomic<double> *leftsum = new atomic<double>[N];
  atomic<int> *knownVars = new atomic<int>[N];
  for (int i = 0; i < N; i++) {
    atomic_init(&(leftsum[i]), 0.0);
    atomic_init(&(knownVars[i]), 0);
  }
  int *colPtr = ldcscMatrix->colPtr;
  int *rowIndices = ldcscMatrix->rowIndices;
  double *values = ldcscMatrix->values;
  
#pragma omp parallel for
  for (int j = 0; j < N; j++) {
    int rowLength = ldrowLengths[j] - 1;
    while (rowLength != knownVars[j]) {
      // spin-wait for all the vars on this row to become known
    }
    
    double xj = (b[j] - leftsum[j]) / values[colPtr[j]];
    x[j] = xj;
    for (int k = colPtr[j] + 1; k < colPtr[j+1]; k++) {
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

void EuroPar16Solver::backwardSolve(double* __restrict b, double* __restrict x) {
  int N = udcscMatrix->N;
  atomic<double> *rightsum = new atomic<double>[N];
  atomic<int> *knownVars = new atomic<int>[N];
  for (int i = 0; i < N; i++) {
    atomic_init(&(rightsum[i]), 0.0);
    atomic_init(&(knownVars[i]), 0);
  }
  int *colPtr = udcscMatrix->colPtr;
  int *rowIndices = udcscMatrix->rowIndices;
  double *values = udcscMatrix->values;
  
#pragma omp parallel for
  for (int j = N - 1; j >= 0; j--) {
    int rowLength = udrowLengths[j] - 1;
    while (rowLength != knownVars[j]) {
      // spin-wait for all the vars on this row to become known
    }
    
    double xj = (b[j] - rightsum[j]) / values[colPtr[j+1]-1];
    x[j] = xj;
    for (int k = colPtr[j+1] - 1 - 1; k >= colPtr[j]; k--) {
      int row = rowIndices[k];
      double mult = values[k] * xj;
      double desired, expected;
      do {
        expected = rightsum[row].load();
        desired = expected + mult;
      } while (!rightsum[row].compare_exchange_weak(expected, desired));
      knownVars[row]++;
    }
  }
  
  delete[] rightsum;
  delete[] knownVars;
}

string EuroPar16Solver::getName() {
  return "EuroPar16";
}
