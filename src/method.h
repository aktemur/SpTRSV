#ifndef _METHOD_H_
#define _METHOD_H_

#include "matrix.h"
#include <string>

#ifdef MKL_EXISTS
#include <mkl.h>
#endif

namespace thundercat {
  class SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads) = 0;

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal. The matrix A should be set beforehand using the init method.
    // This is because some methods prefer CSR format while others like CSC.
    virtual void forwardSolve(double* __restrict b, double* __restrict x) = 0;
    
    virtual std::string getName() = 0;
  };
  
  class ReferenceSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();

  private:
    CSCMatrix *cscMatrix;
  };

  class MKLSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix *csc, int numThreads);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();
    
  private:
#ifdef MKL_EXISTS
    sparse_matrix_t mklL;
    matrix_descr descL;
#endif
  };
}

#endif
