#include "method.h"
#include <iostream>

using namespace thundercat;
using namespace std;

#ifdef MKL_EXISTS
#include <mkl.h>

void MKLSolver::init(CSCMatrix *A, int numThreads) {
  mkl_set_num_threads_local(numThreads);
}

void MKLSolver::forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x) {
  // TODO
}

#else

void MKLSolver::init(CSCMatrix *A, int numThreads) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

void MKLSolver::forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

#endif

string MKLSolver::getName() {
  return "MKL";
}
