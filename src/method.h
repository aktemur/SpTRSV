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
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads, int iters) = 0;

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal. The matrix A should be set beforehand using the init method.
    // This is because some methods prefer CSR format while others like CSC.
    virtual void forwardSolve(double* __restrict b, double* __restrict x) = 0;
    
    virtual std::string getName() = 0;
  };
  
  class SequentialCSRSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads, int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();
    
  private:
    CSRMatrix *csrMatrix;
  };

  class SequentialCSCSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads, int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();

  private:
    CSCMatrix *cscMatrix;
  };

  class EuroPar16Solver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix* csc, int numThreads, int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();
    
  private:
    CSCMatrix *cscMatrix;
    int *rowLengths;
  };

  class MKLSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix *csc, int numThreads, int iters);
    
  protected:
    CSRMatrix *csrMatrix;
    CSCMatrix *cscMatrix;
  };

  class MKLCSRSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLCSCSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLInspectorExecutorSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* csr, CSCMatrix *csc, int numThreads, int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

#ifdef MKL_EXISTS
  protected:
    virtual sparse_matrix_t createMKLMatrix(CSRMatrix* csr, CSCMatrix *csc) = 0;
    
  private:
    sparse_matrix_t mklL;
    matrix_descr descL;
#endif
  };
  
  class MKLInspectorExecutorCSRSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual sparse_matrix_t createMKLMatrix(CSRMatrix* csr, CSCMatrix *csc);
#endif
  };

  class MKLInspectorExecutorCSCSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual sparse_matrix_t createMKLMatrix(CSRMatrix* csr, CSCMatrix *csc);
#endif
  };
}

#endif
