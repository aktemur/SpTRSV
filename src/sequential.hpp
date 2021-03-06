#pragma once

#include "solver.hpp"
#include <cstring>
#include <iostream>

namespace thundercat {
  
template<typename ValueType>
class SequentialCSRSolver: public SparseTriangularSolver<ValueType> {
public:
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    for (int i = 0; i < this->ldcsrMatrix->N; i++) {
      ValueType sum = b[i];
      int k;
      for (k = this->ldcsrMatrix->rowPtr[i]; k < this->ldcsrMatrix->rowPtr[i+1] - 1; k++) {
        int col = this->ldcsrMatrix->colIndices[k];
        sum -= this->ldcsrMatrix->values[k] * x[col];
      }
      x[i] = sum / this->ldcsrMatrix->values[k];
    }
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    for (int i = this->udcsrMatrix->N - 1; i >= 0; i--) {
      ValueType sum = 0.0;
      int k;
      for (k = this->udcsrMatrix->rowPtr[i+1] - 1; k > this->udcsrMatrix->rowPtr[i]; k--) {
        int col = this->udcsrMatrix->colIndices[k];
        sum += this->udcsrMatrix->values[k] * x[col];
      }
      x[i] = (b[i] - sum) / this->udcsrMatrix->values[k];
    }
  }
  
  virtual std::string getName() {
    return "SequentialCSR";
  }
};

template<typename ValueType>
class SequentialCSCSolver: public SparseTriangularSolver<ValueType> {
public:
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);

    leftsum = new ValueType[this->ldcscMatrix->M];
    rightsum = new ValueType[this->udcscMatrix->M];
  }

  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    memset(leftsum, 0, sizeof(ValueType) * this->ldcscMatrix->M);
    
    for (int j = 0; j < this->ldcscMatrix->M; j++) {
      x[j] = (b[j] - leftsum[j]) / this->ldcscMatrix->values[this->ldcscMatrix->colPtr[j]];
      for (int k = this->ldcscMatrix->colPtr[j] + 1; k < this->ldcscMatrix->colPtr[j+1]; k++) {
        int row = this->ldcscMatrix->rowIndices[k];
        leftsum[row] += this->ldcscMatrix->values[k] * x[j];
      }
    }
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    memset(rightsum, 0, sizeof(ValueType) * this->udcscMatrix->M);
    
    for (int j = this->udcscMatrix->M - 1; j >= 0; j--) {
      x[j] = (b[j] - rightsum[j]) / this->udcscMatrix->values[this->udcscMatrix->colPtr[j+1]-1];
      for (int k = this->udcscMatrix->colPtr[j+1] - 1 - 1; k >= this->udcscMatrix->colPtr[j]; k--) {
        int row = this->udcscMatrix->rowIndices[k];
        rightsum[row] += this->udcscMatrix->values[k] * x[j];
      }
    }
  }
  
  virtual std::string getName() {
    return "SequentialCSC";
  }

  virtual ~SequentialCSCSolver() {
    delete[] leftsum;
    delete[] rightsum;
  }
  
private:
  ValueType *leftsum;
  ValueType *rightsum;
};

}

void csrLenForwardSolve(int *rowPtr, int *colIndices, double *values, double *x, double *b, unsigned int N);

namespace thundercat {
template<typename ValueType>
class SequentialCSRLenSolver: public SparseTriangularSolver<ValueType> {
public:
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);

    // Convert the matrix to CSRLen format
    unsigned int i = 0;
    for (; i < ldcsr->N; i++) {
      int length = ldcsr->rowPtr[i + 1] - ldcsr->rowPtr[i];
      length--;
      ldcsr->rowPtr[i] = -(length * 22);
    }
    ldcsr->rowPtr[i] = (5 + 5 + 3 + 3) + (6 + 4 + 7 + 3 + 3);
  }

  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    csrLenForwardSolve(this->ldcsrMatrix->rowPtr, this->ldcsrMatrix->colIndices, this->ldcsrMatrix->values,
                       x, b, this->ldcsrMatrix->N);
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    for (int i = this->udcsrMatrix->N - 1; i >= 0; i--) {
      ValueType sum = 0.0;
      int k;
      for (k = this->udcsrMatrix->rowPtr[i+1] - 1; k > this->udcsrMatrix->rowPtr[i]; k--) {
        int col = this->udcsrMatrix->colIndices[k];
        sum += this->udcsrMatrix->values[k] * x[col];
      }
      x[i] = (b[i] - sum) / this->udcsrMatrix->values[k];
    }
  }
  
  virtual std::string getName() {
    return "SequentialCSRLen";
  }
};
  
} // namespace thundercat
