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
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters) = 0;

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal. The matrix A should be set beforehand using the init method.
    // This is because some methods prefer CSR format while others like CSC.
    virtual void forwardSolve(double* __restrict b, double* __restrict x) = 0;

    virtual void backwardSolve(double* __restrict b, double* __restrict x) = 0;

    virtual std::string getName() = 0;
  };
  
  class SequentialCSRSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();
    
  private:
    CSRMatrix *ldcsrMatrix;
    CSRMatrix *udcsrMatrix;
  };

  class SequentialCSCSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();

  private:
    CSCMatrix *ldcscMatrix;
    CSCMatrix *udcscMatrix;
  };

  class EuroPar16Solver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();
    
  private:
    CSCMatrix *ldcscMatrix;
    CSCMatrix *udcscMatrix;
    int *ldrowLengths;
    int *udrowLengths;
  };

  class MKLSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters);
    
  protected:
    CSRMatrix *ldcsrMatrix;
    CSCMatrix *ldcscMatrix;
    CSRMatrix *udcsrMatrix;
    CSCMatrix *udcscMatrix;
  };

  class MKLCSRSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLCSCSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLInspectorExecutorSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int numThreads, int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc) = 0;
    
  protected:
    sparse_matrix_t mklL;
    matrix_descr descL;
    sparse_matrix_t mklU;
    matrix_descr descU;
#endif
  };
  
  class MKLInspectorExecutorCSRSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc);
#endif
  };

  // NOTE: This method fails at the mkl_sparse_set_sv_hint step
  // by giving error code 6. We still keep it for completeness.
  class MKLInspectorExecutorCSCSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc);
#endif
  };
}

#endif
