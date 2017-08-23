#include "method.h"
#include <iostream>

using namespace thundercat;
using namespace std;

#ifdef MKL_EXISTS

void MKLSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                     CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  mkl_set_num_threads_local(numThreads);
  ldcsrMatrix = ldcsr;
  ldcscMatrix = ldcsc;
  udcsrMatrix = udcsr;
  udcscMatrix = udcsc;
}

void MKLInspectorExecutorSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                      CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  mkl_set_num_threads_local(numThreads);
  sparse_status_t stat;
  
  createMKLMatrices(ldcsr, ldcsc, udcsr, udcsc);
  
  stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_NON_TRANSPOSE, descL, iters);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to set mklL sv hint. Error code: " << stat << "\n";
    exit(1);
  }
  stat = mkl_sparse_optimize(mklL);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to sparse optimize mklL. Error code: " << stat << "\n";
    exit(1);
  }
  stat = mkl_sparse_set_sv_hint(mklU, SPARSE_OPERATION_NON_TRANSPOSE, descU, iters);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to set mklU sv hint. Error code: " << stat << "\n";
    exit(1);
  }
  stat = mkl_sparse_optimize(mklU);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to sparse optimize mklU. Error code: " << stat << "\n";
    exit(1);
  }
}

void MKLInspectorExecutorCSRSolver::createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                                      CSRMatrix *udcsr, CSCMatrix *udcsc) {
  sparse_status_t stat;
  stat = mkl_sparse_d_create_csr(&mklL,
                                 SPARSE_INDEX_BASE_ZERO, ldcsr->N, ldcsr->M,
                                 ldcsr->rowPtr, ldcsr->rowPtr + 1,
                                 ldcsr->colIndices, ldcsr->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create mklL CSR matrix. Error code: " << stat << "\n";
    exit(1);
  }
  descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descL.mode = SPARSE_FILL_MODE_LOWER;
  descL.diag = SPARSE_DIAG_NON_UNIT;
  
  stat = mkl_sparse_d_create_csr(&mklU,
                                 SPARSE_INDEX_BASE_ZERO, udcsr->N, udcsr->M,
                                 udcsr->rowPtr, udcsr->rowPtr + 1,
                                 udcsr->colIndices, udcsr->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create mklU CSR matrix. Error code: " << stat << "\n";
    exit(1);
  }
  descU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descU.mode = SPARSE_FILL_MODE_UPPER;
  descU.diag = SPARSE_DIAG_NON_UNIT;
}

void MKLInspectorExecutorCSCSolver::createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                                      CSRMatrix *udcsr, CSCMatrix *udcsc) {
  sparse_status_t stat;
  stat = mkl_sparse_d_create_csc(&mklL,
                                 SPARSE_INDEX_BASE_ZERO, ldcsc->N, ldcsc->M,
                                 ldcsc->colPtr, ldcsc->colPtr + 1,
                                 ldcsc->rowIndices, ldcsc->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create mklL CSC matrix. Error code: " << stat << "\n";
    exit(1);
  }
  descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descL.mode = SPARSE_FILL_MODE_LOWER;
  descL.diag = SPARSE_DIAG_NON_UNIT;

  stat = mkl_sparse_d_create_csc(&mklU,
                                 SPARSE_INDEX_BASE_ZERO, udcsc->N, udcsc->M,
                                 udcsc->colPtr, udcsc->colPtr + 1,
                                 udcsc->rowIndices, udcsc->values);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to create mklU CSC matrix. Error code: " << stat << "\n";
    exit(1);
  }
  descU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descU.mode = SPARSE_FILL_MODE_UPPER;
  descU.diag = SPARSE_DIAG_NON_UNIT;
}

void MKLCSRSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = ldcsrMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = ldcsrMatrix->values;
  const int *indx = ldcsrMatrix->colIndices;
  const int *pntrb = ldcsrMatrix->rowPtr;
  const int *pntre = ldcsrMatrix->rowPtr + 1;
  
  mkl_dcsrsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLCSCSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = ldcscMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = ldcscMatrix->values;
  const int *indx = ldcscMatrix->rowIndices;
  const int *pntrb = ldcscMatrix->colPtr;
  const int *pntre = ldcscMatrix->colPtr + 1;
  
  mkl_dcscsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLCSRSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = udcsrMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TUNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = udcsrMatrix->values;
  const int *indx = udcsrMatrix->colIndices;
  const int *pntrb = udcsrMatrix->rowPtr;
  const int *pntre = udcsrMatrix->rowPtr + 1;
  
  mkl_dcsrsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLCSCSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  const char transa = 'n';
  const int numColumns = udcscMatrix->M;
  const double alpha = 1.0;
  const char matdescr[] = "TUNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
  const double *val = udcscMatrix->values;
  const int *indx = udcscMatrix->rowIndices;
  const int *pntrb = udcscMatrix->colPtr;
  const int *pntre = udcscMatrix->colPtr + 1;
  
  mkl_dcscsv(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
}

void MKLInspectorExecutorSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  sparse_status_t stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklL, descL, b, x);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to forward solve. Error code: " << stat << "\n";
    exit(1);
  }
}

void MKLInspectorExecutorSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  sparse_status_t stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklU, descU, b, x);
  if (SPARSE_STATUS_SUCCESS != stat) {
    cerr << "Failed to backward solve. Error code: " << stat << "\n";
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
