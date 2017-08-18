#include "method.h"
#include <iostream>

using namespace thundercat;
using namespace std;

#ifdef MKL_EXISTS

void MKLSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  mkl_set_num_threads_local(numThreads);
  csrMatrix = csr;
  cscMatrix = csc;
}

void MKLInspectorExecutorSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads, int iters) {
  mkl_set_num_threads_local(numThreads);
  sparse_status_t stat;
  
  mklL = createMKLMatrix(csr, csc);
  
  descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descL.mode = SPARSE_FILL_MODE_LOWER;
  descL.diag = SPARSE_DIAG_NON_UNIT;
  
  stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_NON_TRANSPOSE, descL, iters);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to set MKL sv hint. Error code: " << stat << "\n";
    exit(1);
  }
  
  stat = mkl_sparse_optimize(mklL);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to sparse optimize. Error code: " << stat << "\n";
    exit(1);
  }
}

sparse_matrix_t MKLInspectorExecutorCSRSolver::createMKLMatrix(CSRMatrix *csr, CSCMatrix *csc) {
  sparse_matrix_t mklA;
  sparse_status_t stat = mkl_sparse_d_create_csr(&mklA,
                                                 SPARSE_INDEX_BASE_ZERO, csr->N, csr->M,
                                                 csr->rowPtr, csr->rowPtr + 1,
                                                 csr->colIndices, csr->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create MKL CSR matrix. Error code: " << stat << "\n";
    exit(1);
  }
  return mklA;
}

sparse_matrix_t MKLInspectorExecutorCSCSolver::createMKLMatrix(CSRMatrix *csr, CSCMatrix *csc) {
  sparse_matrix_t mklA;
  sparse_status_t stat = mkl_sparse_d_create_csc(&mklA,
                                                 SPARSE_INDEX_BASE_ZERO, csc->N, csc->M,
                                                 csc->colPtr, csc->colPtr + 1,
                                                 csc->rowIndices, csc->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create MKL CSC matrix. Error code: " << stat << "\n";
    exit(1);
  }
  return mklA;
}

void MKLCSRSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = csrMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = csrMatrix->values;
  const int *indx = csrMatrix->colIndices;
  const int *pntrb = csrMatrix->rowPtr;
  const int *pntre = csrMatrix->rowPtr + 1;
  
  mkl_dcsrsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLCSCSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = cscMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = cscMatrix->values;
  const int *indx = cscMatrix->rowIndices;
  const int *pntrb = cscMatrix->colPtr;
  const int *pntre = cscMatrix->colPtr + 1;
  
  mkl_dcscsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLInspectorExecutorSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  sparse_status_t stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklL, descL, b, x);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to solve. Error code: " << stat << "\n";
    exit(1);
  }
}

#else

void MKLSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

void MKLInspectorExecutorSolver::init(CSRMatrix *csr, CSCMatrix *csc, int numThreads) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

void MKLCSRSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

void MKLCSCSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

void MKLInspectorExecutorSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  cerr << "MKL is not supported on this platform.\n";
  exit(1);
}

#endif

string MKLCSRSolver::getName() {
  return "MKL-CSR";
}

string MKLCSCSolver::getName() {
  return "MKL-CSC";
}

string MKLInspectorExecutorCSRSolver::getName() {
  return "MKL-IE-CSR";
}

string MKLInspectorExecutorCSCSolver::getName() {
  return "MKL-IE-CSC";
}
