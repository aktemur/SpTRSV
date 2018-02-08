#pragma once 

#include "solver.hpp"
#include <cstring>
#include <iostream>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include <queue>
#include <deque>
#include "concurrentqueue.h"

namespace thundercat {

#define PACKSIZE 1
  
template<typename ValueType>
class ExperimentalSolver: public SparseTriangularSolver<ValueType> {
public:
  ExperimentalSolver() : taskQueue() { }
  
  virtual void init(CSRMatrix<ValueType> *ldcsr, CSCMatrix<ValueType> *ldcsc,
                    CSRMatrix<ValueType> *udcsr, CSCMatrix<ValueType> *udcsc,
                    int iters) {
    SparseTriangularSolver<ValueType>::init(ldcsr, ldcsc, udcsr, udcsc, iters);
    
    const int N = ldcsc->N;
    unknownVars = new int[N / PACKSIZE];
    rowsToSolve.resize(omp_get_max_threads());
    
    int *levels = new int[N / PACKSIZE];
    int maxLevel = 0;
    
    memset(unknownVars, 0, sizeof(int) * (N / PACKSIZE));
    memset(levels, 0, sizeof(int) * (N / PACKSIZE));
    
    for (int i = 0; i < N; i++) {
      const int pack = i / PACKSIZE;
      const int length = ldcsr->rowPtr[i + 1] - ldcsr->rowPtr[i];
      int outerDependencies = length - 1;
      for (int k = ldcsc->colPtr[i] + 1; k < ldcsc->colPtr[i+1]; k++) {
        const int row = ldcsc->rowIndices[k];
        const int dependentPack = row / PACKSIZE;
        if (dependentPack == pack) {
          outerDependencies--;
        } else {
          if (levels[dependentPack] < levels[pack] + 1) {
            levels[dependentPack] = levels[pack] + 1;
          }
        }
      }
      unknownVars[pack] += outerDependencies;
      if (maxLevel < levels[pack]) {
        maxLevel = levels[pack];
      }
    }
    
    dependencies = new std::atomic<int>[N / PACKSIZE];
    
    dependencyGraph.resize(maxLevel + 1);
    for (int pack = 0; pack < N / PACKSIZE; pack++) {
      const int level = levels[pack];
      dependencyGraph[level].push_back(pack);
    }
    delete[] levels;
  }
  
  virtual void forwardSolve(ValueType* __restrict b, ValueType* __restrict x) {
    CSCMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;

    const int N = ldcscMatrix->N;
    assert(sizeof(int) == sizeof(std::atomic<int>));
    memcpy(dependencies, unknownVars, sizeof(std::atomic<int>) * (N / PACKSIZE));
    
    levels(b, x);
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
    return "Experimental";
  }
  
private:
  void spinWait(ValueType* __restrict b, ValueType* __restrict x) {
    CSRMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    CSRMatrix<ValueType> *ldcsrMatrix = this->ldcsrMatrix;
    
    const int N = ldcscMatrix->N;
    
    #pragma omp parallel for
    for (int pack = 0; pack < N / PACKSIZE; pack++) {
      while (dependencies[pack] > 0) {
        // spin wait
      }
      const int packBegin = pack * PACKSIZE;
      for (int p = 0; p < PACKSIZE; p++) {
        const int i = packBegin + p;
        ValueType leftsum = 0;
        int j;
        for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
          int col = ldcsrMatrix->colIndices[j];
          leftsum += ldcsrMatrix->values[j] * x[col];
        }
        ValueType xi = (b[i] - leftsum) / ldcsrMatrix->values[j];
        x[i] = xi;
        for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
          const int row = ldcscMatrix->rowIndices[k];
          const int dependentPack = row / PACKSIZE;
          //printf("k %d %d %d\n", i, row, dependentPack);
          if (dependentPack != pack)
            (dependencies[dependentPack])--;
        }
      }
    }
  }
  
  void localQueue(ValueType* __restrict b, ValueType* __restrict x) {
    CSRMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    CSRMatrix<ValueType> *ldcsrMatrix = this->ldcsrMatrix;
    
    const int N = ldcscMatrix->N;
    
    #pragma omp parallel for
    for (int pack = 0; pack < (N / PACKSIZE); pack++) {
      const int threadId = omp_get_thread_num();
      if (dependencies[pack] == 0) {
        rowsToSolve[threadId].push_back(pack);
      }
    }
    /*
    const int threadCount = omp_get_max_threads();
    int threadId = 0;
    for (int pack = 0; pack < (N / PACKSIZE); pack++) {
      if (dependencies[pack] == 0) {
        rowsToSolve[threadId].push_back(pack);
        threadId = (threadId + 1) % threadCount;
      }
    }
    */
    
    #pragma omp parallel
    {
      const int threadId = omp_get_thread_num();
      while (!rowsToSolve[threadId].empty()) {
        const int pack = rowsToSolve[threadId].front();
        rowsToSolve[threadId].pop_front();
        const int packBegin = pack * PACKSIZE;
        for (int p = 0; p < PACKSIZE; p++) {
          const int i = packBegin + p;
          ValueType leftsum = 0;
          int j;
          for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
            const int col = ldcsrMatrix->colIndices[j];
            leftsum += ldcsrMatrix->values[j] * x[col];
          }
          x[i] = (b[i] - leftsum) / ldcsrMatrix->values[j];
          for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
            const int row = ldcscMatrix->rowIndices[k];
            const int dependentPack = row / PACKSIZE;
            if (dependentPack != pack && --(dependencies[dependentPack]) == 0) {
              rowsToSolve[threadId].push_back(dependentPack);
            }
          }
        }
      }
    }
  }
  
  void sharedQueue(ValueType* __restrict b, ValueType* __restrict x) {
    CSRMatrix<ValueType> *ldcscMatrix = this->ldcscMatrix;
    CSRMatrix<ValueType> *ldcsrMatrix = this->ldcsrMatrix;
    std::atomic<int> solved(0);
    const int N = ldcscMatrix->N;
    
    #pragma omp parallel for
    for (int pack = 0; pack < (N / PACKSIZE); pack++) {
      if (dependencies[pack] == 0) {
        taskQueue.enqueue(pack);
      }
    }
    
    #pragma omp parallel
    {
      int pack = -1;
      while (solved < (N / PACKSIZE)) {
        //while (taskQueue.size_approx() > 0) {
        if (taskQueue.try_dequeue(pack)) {
          solved++;
          const int packBegin = pack * PACKSIZE;
          for (int p = 0; p < PACKSIZE; p++) {
            const int i = packBegin + p;
            ValueType leftsum = 0;
            int j;
            for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
              const int col = ldcsrMatrix->colIndices[j];
              leftsum += ldcsrMatrix->values[j] * x[col];
            }
            x[i] = (b[i] - leftsum) / ldcsrMatrix->values[j];
            for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
              const int row = ldcscMatrix->rowIndices[k];
              const int dependentPack = row / PACKSIZE;
              if (dependentPack != pack && --(dependencies[dependentPack]) == 0) {
                taskQueue.enqueue(dependentPack);
              }
            }
          }
        }
      }
    }
  }
  
  void levels(ValueType* __restrict b, ValueType* __restrict x) {
    CSRMatrix<ValueType> *ldcsrMatrix = this->ldcsrMatrix;

    for (int level = 0; level < dependencyGraph.size(); level++) {
      #pragma omp parallel for
      for (int k = 0; k < dependencyGraph[level].size(); k++) {
        const int pack = dependencyGraph[level][k];
        const int packBegin = pack * PACKSIZE;
        for (int p = 0; p < PACKSIZE; p++) {
          const int i = packBegin + p;
          ValueType leftsum = 0;
          int j;
          for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
            const int col = ldcsrMatrix->colIndices[j];
            leftsum += ldcsrMatrix->values[j] * x[col];
          }
          x[i] = (b[i] - leftsum) / ldcsrMatrix->values[j];
        }
      }
    }
  }

private:
  int *unknownVars;
  std::atomic<int> *dependencies;
  std::vector<std::deque<int> > rowsToSolve;
  moodycamel::ConcurrentQueue<int> taskQueue;
  std::vector< std::vector<int> > dependencyGraph;  
};

} // namespace thundercat

