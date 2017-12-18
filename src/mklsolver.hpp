#pragma once

#include "solver.hpp"
#include <cstring>
#include <atomic>
#include <iostream>
#ifdef MKL_EXISTS
#include <mkl.h>
#endif

extern bool DEBUG_MODE_ON;

namespace thundercat {
#ifdef MKL_EXISTS
  //
  // Template functions for MKL's _s_ and _d_ functions.
  //
  template<typename ValueType>
  void mkl_csrsv(const char *transa, const MKL_INT *m, const ValueType *alpha, const char *matdescra, const ValueType *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const ValueType *x, ValueType *y);
  
  template<>
  void mkl_csrsv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y) {
    mkl_scsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y);
  }
  
  template<>
  void mkl_csrsv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y) {
    mkl_dcsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y);
  }
  
  template<typename ValueType>
  void mkl_cscsv(const char *transa, const MKL_INT *m, const ValueType *alpha, const char *matdescra, const ValueType *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const ValueType *x, ValueType *y);
  
  template<>
  void mkl_cscsv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y) {
    mkl_scscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y);
  }
  
  template<>
  void mkl_cscsv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y) {
    mkl_dcscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y);
  }
  
  template<typename ValueType>
  sparse_status_t mkl_sparse_trsv(sparse_operation_t operation, ValueType alpha, const sparse_matrix_t A, struct matrix_descr descr, const ValueType *x, ValueType *y);
  
  template<>
  sparse_status_t mkl_sparse_trsv(sparse_operation_t operation, float alpha, const sparse_matrix_t A, struct matrix_descr descr, const float *x, float *y) {
    mkl_sparse_s_trsv(operation, alpha, A, descr, x, y);
  }
  
  template<>
  sparse_status_t mkl_sparse_trsv(sparse_operation_t operation, double alpha, const sparse_matrix_t A, struct matrix_descr descr, const double *x, double *y) {
    mkl_sparse_d_trsv(operation, alpha, A, descr, x, y);
  }
  
  template<typename ValueType>
  sparse_status_t mkl_sparse_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, ValueType *values);
  
  template<>
  sparse_status_t mkl_sparse_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values) {
    mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
  
  template<>
  sparse_status_t mkl_sparse_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values) {
    mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
  
  template<typename ValueType>
  sparse_status_t mkl_sparse_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, ValueType *values);
  
  template<>
  sparse_status_t mkl_sparse_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values) {
    mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
  
  template<>
  sparse_status_t mkl_sparse_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values) {
    mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
#endif

//
// MKLSolver
//
template<typename ValueType>
class MKLSolver: public SparseTriangularSolver<ValueType> {
public:
  virtual void initThreads(int numThreads) {
#ifdef MKL_EXISTS
    mkl_set_num_threads_local(numThreads);
    if (DEBUG_MODE_ON)
      std::cout << "NumThreads: " << mkl_get_max_threads() << "\n";
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
};

//
// MKLCSRSolver
//
template<typename ValueType>
class MKLCSRSolver: public MKLSolver<ValueType> {
public:
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    CSRMatrix<ValueType> *ldcsrMatrix = this->ldcsrMatrix;
    const char transa = 'n';
    const int numColumns = ldcsrMatrix->M;
    const ValueType alpha = 1.0;
    const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
    const ValueType *val = ldcsrMatrix->values;
    const int *indx = ldcsrMatrix->colIndices;
    const int *pntrb = ldcsrMatrix->rowPtr;
    const int *pntre = ldcsrMatrix->rowPtr + 1;
    
    mkl_csrsv<ValueType>(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    CSRMatrix<ValueType> *udcsrMatrix = this->udcsrMatrix;
    const char transa = 'n';
    const int numColumns = udcsrMatrix->M;
    const ValueType alpha = 1.0;
    const char matdescr[] = "TUNC__"; // Triangular, Upper, Non-unit diagonal, C-based indexing
    const ValueType *val = udcsrMatrix->values;
    const int *indx = udcsrMatrix->colIndices;
    const int *pntrb = udcsrMatrix->rowPtr;
    const int *pntre = udcsrMatrix->rowPtr + 1;
    
    mkl_csrsv<ValueType>(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }

  virtual std::string getName() {
    return "MKL-CSR";
  }
};

//
// MKLCSCSolver
//
template<typename ValueType>
class MKLCSCSolver: public MKLSolver<ValueType> {
public:
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    CSCMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    const char transa = 'n';
    const int numColumns = ldcscMatrix->M;
    const ValueType alpha = 1.0;
    const char matdescr[] = "TLNC__"; // Triangular, Lower, Non-unit diagonal, C-based indexing
    const ValueType *val = ldcscMatrix->values;
    const int *indx = ldcscMatrix->rowIndices;
    const int *pntrb = ldcscMatrix->colPtr;
    const int *pntre = ldcscMatrix->colPtr + 1;
    
    mkl_cscsv<ValueType>(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    CSCMatrix<ValueType> *udcscMatrix = this->udcscMatrix;
    const char transa = 'n';
    const int numColumns = udcscMatrix->M;
    const ValueType alpha = 1.0;
    const char matdescr[] = "TUNC__"; // Triangular, Upper, Non-unit diagonal, C-based indexing
    const ValueType *val = udcscMatrix->values;
    const int *indx = udcscMatrix->rowIndices;
    const int *pntrb = udcscMatrix->colPtr;
    const int *pntre = udcscMatrix->colPtr + 1;
    
    mkl_cscsv<ValueType>(&transa, &numColumns, &alpha, matdescr, val, indx, pntrb, pntre, b, x);
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }

  virtual std::string getName() {
    return "MKL-CSC";
  }
};

//
// MKLIESolver
//
template<typename ValueType>
class MKLInspectorExecutorSolver: public SparseTriangularSolver<ValueType> {
public:
  virtual void initThreads(int numThreads) {
#ifdef MKL_EXISTS
    mkl_set_num_threads_local(numThreads);
    if (DEBUG_MODE_ON)
      std::cout << "NumThreads: " << mkl_get_max_threads() << "\n";
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
#ifdef MKL_EXISTS
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);
    
    sparse_status_t stat;
    createMKLMatrices(ldcsr, ldcsc, udcsr, udcsc);
    
    stat = mkl_sparse_set_sv_hint(mklL, SPARSE_OPERATION_NON_TRANSPOSE, descL, iters);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to set mklL sv hint. Error code: " << stat << "\n";
      exit(1);
    }
    stat = mkl_sparse_optimize(mklL);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to sparse optimize mklL. Error code: " << stat << "\n";
      exit(1);
    }
    stat = mkl_sparse_set_sv_hint(mklU, SPARSE_OPERATION_NON_TRANSPOSE, descU, iters);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to set mklU sv hint. Error code: " << stat << "\n";
      exit(1);
    }
    stat = mkl_sparse_optimize(mklU);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to sparse optimize mklU. Error code: " << stat << "\n";
      exit(1);
    }
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    sparse_status_t stat = mkl_sparse_trsv<ValueType>(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklL, descL, b, x);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to forward solve. Error code: " << stat << "\n";
      exit(1);
    }
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
#ifdef MKL_EXISTS
    sparse_status_t stat = mkl_sparse_trsv<ValueType>(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklU, descU, b, x);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to backward solve. Error code: " << stat << "\n";
      exit(1);
    }
#else
    std::cerr << "MKL is not supported on this platform.\n";
    exit(1);
#endif  
  }
  
#ifdef MKL_EXISTS
protected:
  virtual void createMKLMatrices(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                                 CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc) = 0;
  
protected:
  sparse_matrix_t mklL;
  matrix_descr descL;
  sparse_matrix_t mklU;
  matrix_descr descU;
#endif
};

//
// MKLIECSRSolver
//
template<typename ValueType>
class MKLInspectorExecutorCSRSolver: public MKLInspectorExecutorSolver<ValueType> {
#ifdef MKL_EXISTS
protected:
  virtual void createMKLMatrices(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                                 CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc) {
    sparse_status_t stat;
    stat = mkl_sparse_create_csr<ValueType>(&(this->mklL),
                                   SPARSE_INDEX_BASE_ZERO, ldcsr->N, ldcsr->M,
                                   ldcsr->rowPtr, ldcsr->rowPtr + 1,
                                   ldcsr->colIndices, ldcsr->values);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to create mklL CSR matrix. Error code: " << stat << "\n";
      exit(1);
    }
    this->descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    this->descL.mode = SPARSE_FILL_MODE_LOWER;
    this->descL.diag = SPARSE_DIAG_NON_UNIT;
    
    stat = mkl_sparse_create_csr<ValueType>(&(this->mklU),
                                   SPARSE_INDEX_BASE_ZERO, udcsr->N, udcsr->M,
                                   udcsr->rowPtr, udcsr->rowPtr + 1,
                                   udcsr->colIndices, udcsr->values);
    if (SPARSE_STATUS_SUCCESS != stat) {
      std::cerr << "Failed to create mklU CSR matrix. Error code: " << stat << "\n";
      exit(1);
    }
    this->descU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    this->descU.mode = SPARSE_FILL_MODE_UPPER;
    this->descU.diag = SPARSE_DIAG_NON_UNIT;
  }
#endif

public:
  virtual std::string getName() {
    return "MKL-IE-CSR";
  }
};

} // namespace thundercat

