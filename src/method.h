#ifndef _METHOD_H_
#define _METHOD_H_

#include "matrix.h"
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

namespace thundercat {
  class SpTRSVMethod {
  public:
    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal.
    virtual void solve(CSCMatrix* A, double* __restrict b, double* __restrict x) = 0;
  };
  
  class ReferenceSolver: public SpTRSVMethod {
  public:
    virtual void solve(CSCMatrix* A, double* __restrict b, double* __restrict x);
  };

  class MKLSolver: public SpTRSVMethod {
  public:
    virtual void solve(CSCMatrix* A, double* __restrict b, double* __restrict x);
  };
}

#endif
