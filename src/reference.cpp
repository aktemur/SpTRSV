#include "method.h"
#include <cstring>

using namespace thundercat;
using namespace std;

void ReferenceSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads) {
  cscMatrix = csc;
}

void ReferenceSolver::forwardSolve(double* __restrict b, double* __restrict x) {
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

string ReferenceSolver::getName() {
  return "Reference";
}
