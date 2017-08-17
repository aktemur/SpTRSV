#include "method.h"
#include <iostream>

using namespace thundercat;
using namespace std;

#ifdef MKL_EXISTS

#define REPEAT 150 //TODO

void MKLSolver::init(CSCMatrix *A, int numThreads) {
  mkl_set_num_threads_local(numThreads);
  
  sparse_matrix_t mklA;
  sparse_status_t stat = mkl_sparse_d_create_csc(&mklA,
                                                 SPARSE_INDEX_BASE_ZERO, A->N, A->M,
                                                 A->colPtr, A->colPtr + 1,
                                                 A->rowIndices, A->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create MKL CSC.\n";
    exit(1);
  }
  
  descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descL.mode = SPARSE_FILL_MODE_LOWER;
  descL.diag = SPARSE_DIAG_NON_UNIT;
  
  stat = mkl_sparse_copy(mklA, descL, &mklL);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create MKL CSC lower.\n";
    exit(1);
  }
  
  stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_NON_TRANSPOSE, descL, REPEAT);
  
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to set MKL sv hint.\n";
    exit(1);
  }
  
  stat = mkl_sparse_optimize(mklL);
  
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to sparse optimize\n";
    exit(1);
  }

}

void MKLSolver::forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x) {
  mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklL, descL, b, x);
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
