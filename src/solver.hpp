#pragma once

#include "matrix.hpp"
#include <string>
#include <iostream>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

extern bool DEBUG_MODE_ON;

namespace thundercat {
#ifndef OPENMP_EXISTS
  int omp_get_thread_num() {
    return 0;
  }
  int omp_get_max_threads() {
    return 1;
  }
#endif

  template<typename ValueType>
  class SparseTriangularSolver {
  public:
    virtual void initThreads(int numThreads) {
#ifdef OPENMP_EXISTS
      omp_set_num_threads(numThreads);
      int nthreads = -1;
#pragma omp parallel
      {
#pragma omp master
        {
          nthreads = omp_get_num_threads();
        }
      }
      if (DEBUG_MODE_ON)
        std::cout << "NumThreads: " << nthreads << "\n";
#else
      if (DEBUG_MODE_ON)
        std::cout << "OpenMP not available.\n";
#endif
    }

    virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                      CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                      int iters) {
      ldcsrMatrix = ldcsr;
      ldcscMatrix = ldcsc;
      udcsrMatrix = udcsr;
      udcscMatrix = udcsc;
    }

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal. The matrix A should be set beforehand using the init method.
    // This is because some methods prefer CSR format while others like CSC.
    virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) = 0;

    virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) = 0;

    virtual std::string getName() = 0;
    
  protected:
    CSRMatrix<ValueType> *ldcsrMatrix;
    CSCMatrix<ValueType> *ldcscMatrix;
    CSRMatrix<ValueType> *udcsrMatrix;
    CSCMatrix<ValueType> *udcscMatrix;
  };
}
