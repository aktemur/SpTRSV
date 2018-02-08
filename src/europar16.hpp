#pragma once

#include "solver.hpp"
#include <cstring>
#include <atomic>

/**
 * This solver is adapted from 
 *   Liu W., Li A., Hogg J., Duff I.S., Vinter B. (2016)
 *   A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves.
 *   In Euro-Par 2016: Parallel Processing.
 */

namespace thundercat {

template<typename ValueType>
class EuroPar16Solver: public SparseTriangularSolver<ValueType> {
public:
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);
    ldrowLengths = new int[ldcsc->N];
    udrowLengths = new int[udcsc->N];
    
    #pragma omp parallel for
    for (int i = 0; i < ldcsc->NZ; i++) {
      #pragma omp atomic update
      ldrowLengths[ldcsc->rowIndices[i]]++;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < udcsc->NZ; i++) {
      #pragma omp atomic update
      udrowLengths[udcsc->rowIndices[i]]++;
    }
  }
  
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    CSCMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    int N = ldcscMatrix->N;
    std::atomic<ValueType> *leftsum = new std::atomic<ValueType>[N];
    std::atomic<int> *knownVars = new std::atomic<int>[N];
    for (int i = 0; i < N; i++) {
      atomic_init(&(leftsum[i]), 0.0);
      atomic_init(&(knownVars[i]), 0);
    }
    int *colPtr = ldcscMatrix->colPtr;
    int *rowIndices = ldcscMatrix->rowIndices;
    ValueType *values = ldcscMatrix->values;
    
    #pragma omp parallel for
    for (int j = 0; j < N; j++) {
      int rowLength = ldrowLengths[j] - 1;
      while (rowLength != knownVars[j]) {
        // spin-wait for all the vars on this row to become known
      }
      
      ValueType xj = (b[j] - leftsum[j]) / values[colPtr[j]];
      x[j] = xj;
      for (int k = colPtr[j] + 1; k < colPtr[j+1]; k++) {
        int row = rowIndices[k];
        ValueType mult = values[k] * xj;
        ValueType desired, expected;
        do {
          expected = leftsum[row].load();
          desired = expected + mult;
        } while (!leftsum[row].compare_exchange_weak(expected, desired));
        knownVars[row]++;
      }
    }
    
    delete[] leftsum;
    delete[] knownVars;
  }
  
  virtual void backwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    CSCMatrix<ValueType> *udcscMatrix = this->udcscMatrix;
    int N = udcscMatrix->N;
    std::atomic<ValueType> *rightsum = new std::atomic<ValueType>[N];
    std::atomic<int> *knownVars = new std::atomic<int>[N];
    for (int i = 0; i < N; i++) {
      atomic_init(&(rightsum[i]), 0.0);
      atomic_init(&(knownVars[i]), 0);
    }
    int *colPtr = udcscMatrix->colPtr;
    int *rowIndices = udcscMatrix->rowIndices;
    ValueType *values = udcscMatrix->values;
    
    #pragma omp parallel for
    for (int j = N - 1; j >= 0; j--) {
      int rowLength = udrowLengths[j] - 1;
      while (rowLength != knownVars[j]) {
        // spin-wait for all the vars on this row to become known
      }
      
      ValueType xj = (b[j] - rightsum[j]) / values[colPtr[j+1]-1];
      x[j] = xj;
      for (int k = colPtr[j+1] - 1 - 1; k >= colPtr[j]; k--) {
        int row = rowIndices[k];
        ValueType mult = values[k] * xj;
        ValueType desired, expected;
        do {
          expected = rightsum[row].load();
          desired = expected + mult;
        } while (!rightsum[row].compare_exchange_weak(expected, desired));
        knownVars[row]++;
      }
    }
    
    delete[] rightsum;
    delete[] knownVars;
  }
  
  virtual std::string getName() {
    return "EuroPar16";
  }
  
private:
  int *ldrowLengths;
  int *udrowLengths;
};

} // namespace thundercat
