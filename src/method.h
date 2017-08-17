#ifndef _METHOD_H_
#define _METHOD_H_

#include "matrix.h"
#include <string>

namespace thundercat {
  class SparseTriangularSolver {
  public:
    // Optional initialization step in case the method wants to do some pre computation
    virtual void init(CSCMatrix* A, int numThreads);

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal.
    virtual void forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x) = 0;
    
    virtual std::string getName() = 0;
  };
  
  class ReferenceSolver: public SparseTriangularSolver {
  public:
    virtual void forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x);

    virtual std::string getName();
  };

  class MKLSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSCMatrix* A, int numThreads);

    virtual void forwardSolve(CSCMatrix* A, double* __restrict b, double* __restrict x);

    virtual std::string getName();
  };
}

#endif
