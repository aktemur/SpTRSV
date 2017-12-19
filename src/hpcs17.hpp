#pragma once 

#include "solver.hpp"

namespace thundercat {

template<typename ValueType>
class HPCS17Solver: public SparseTriangularSolver<ValueType> {
public:
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);
    
    const int NZ = ldcsc->NZ;
    
    int *levels = new int[NZ];
    int maxLevel = 0;
    
    memset(levels, 0, sizeof(int) * NZ);
    
    for (int i = 0; i < NZ; i++) {
      // TODO
    }
    
    delete[] levels;
  }
  
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    CSCMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    // TODO
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    CSRMatrix<ValueType> *udcsrMatrix = this->udcsrMatrix;
    
    // No parallelization yet.
    for (int i = udcsrMatrix->N - 1; i >= 0; i--) {
      ValueType sum = 0.0;
      int k;
      for (k = udcsrMatrix->rowPtr[i+1] - 1; k > udcsrMatrix->rowPtr[i]; k--) {
        int col = udcsrMatrix->colIndices[k];
        sum += udcsrMatrix->values[k] * x[col];
      }
      x[i] = (b[i] - sum) / udcsrMatrix->values[k];
    }
  }
  
  virtual std::string getName() {
    return "HPCS17";
  }
  
private:
  std::vector< std::vector<int> > dependencyGraph;
};

} // namespace thundercat

