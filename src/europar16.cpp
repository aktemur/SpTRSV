#include "method.h"
#include <cstring>

using namespace thundercat;
using namespace std;

/**
 * This solver is taken from 
 *   Liu W., Li A., Hogg J., Duff I.S., Vinter B. (2016)
 *   A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves.
 *   In Euro-Par 2016: Parallel Processing.
 */

void EuroPar16Solver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  cscMatrix = csc;
  rowLengths = new int[cscMatrix->N];
  // TODO: Do this in parallel
  for (int i = 0; i < cscMatrix->NZ; i++) {
    rowLengths[cscMatrix->rowIndices[i]]++; // TODO: This should be atomic
  }
}

void EuroPar16Solver::forwardSolve(double* __restrict b, double* __restrict x) {
  int N = cscMatrix->N;
  double *leftsum = new double[N];
  memset(leftsum, 0, sizeof(double) * N);
  int *knownVars = new int[N];
  memset(knownVars, 0, sizeof(int) * N);
  int *colPtr = cscMatrix->colPtr;
  int *rowIndices = cscMatrix->rowIndices;
  double *values = cscMatrix->values;
  
  for (int j = 0; j < N; j++) {
    int rowLength = rowLengths[j] - 1;
    while (rowLength != knownVars[j]) {
      // spin-wait for all the vars on this row to become known
    }
    
    double xj = (b[j] - leftsum[j]) / values[colPtr[j]];
    x[j] = xj;
    for (int k = colPtr[j] + 1; k < cscMatrix->colPtr[j+1]; k++) {
      int row = rowIndices[k];
      leftsum[row] += values[k] * xj; // TODO: This should be atomic
      knownVars[row]++;               // TODO: This should be atomic
    }
  }
  
  delete[] leftsum;
  delete[] knownVars;
}

string EuroPar16Solver::getName() {
  return "EuroPar16";
}
