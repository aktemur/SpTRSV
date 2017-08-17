#include "method.h"

using namespace thundercat;
using namespace std;

void ReferenceSolver::forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x) {
  double *leftsum = new double[A->M];
  memset(leftsum, 0, sizeof(double) * A->M);
  
  for (int j = 0; j < A->M; j++) {
    x[j] = (b[j] - leftsum[j]) / A->values[A->colPtr[j]];
    for (int k = A->colPtr[j] + 1; k < A->colPtr[j+1]; k++) {
      int row = A->rowIndices[k];
      leftsum[row] += A->values[k] * x[j];
    }
  }
  
  delete[] leftsum;
}

string ReferenceSolver::getName() {
  return "Reference";
}
