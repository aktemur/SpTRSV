#include "method.h"
#include <cstring>
#include <iostream>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif
#include <tbb/atomic.h>

using namespace thundercat;
using namespace std;

void ParallelCSCSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                             CSRMatrix *udcsr, CSCMatrix *udcsc, int iters) {
  ldcscMatrix = ldcsc;
  ldcsrMatrix = ldcsr;
  udcscMatrix = udcsc;
  initialDependencies = new int[ldcscMatrix->M];

  for (int i = 0; i < ldcscMatrix->M; i++) {
    initialDependencies[i] = (ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i]) - 1;
  }

}

void ParallelCSCSolver::backwardSolve(double *__restrict b, double *__restrict x) {
  double *rightsum = new double[udcscMatrix->M];
  memset(rightsum, 0, sizeof(double) * udcscMatrix->M);
  for (int j = udcscMatrix->M - 1; j >= 0; j--) {
    x[j] = (b[j] - rightsum[j]) / udcscMatrix->values[udcscMatrix->colPtr[j + 1] - 1];
    for (int k = udcscMatrix->colPtr[j + 1] - 1 - 1; k >= udcscMatrix->colPtr[j]; k--) {
      int row = udcscMatrix->rowIndices[k];
      rightsum[row] += udcscMatrix->values[k] * x[j];
    }
  }

  delete[] rightsum;
}


/**
 * CSC Solver with TBB synchronisation primitives
 */
TBBSolver::TBBSolver():indexQueue() {}

void TBBSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  tbb::atomic<int> solved;
  solved = 0;
  tbb::atomic<int> *dependencies = new tbb::atomic<int>[ldcscMatrix->M];

  #pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
    dependencies[i] = initialDependencies[i];
    if (initialDependencies[i] == 0) {
      indexQueue.push(i);
    }
  }

  #pragma omp parallel
  while (solved < ldcscMatrix->M) {

    int solve_index = -1;

    if (indexQueue.try_pop(solve_index)) {

      double componentSum = 0.0;
      for (
          int componentIndex = ldcsrMatrix->rowPtr[solve_index];
          componentIndex < ldcsrMatrix->rowPtr[solve_index + 1] - 1;
          componentIndex++) {
        componentSum +=
            x[ldcsrMatrix->colIndices[componentIndex]] * ldcsrMatrix->values[componentIndex];

      }
      x[solve_index] = (b[solve_index] - componentSum) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];
        int ready;
        ready = --(dependencies[ldcscMatrix->rowIndices[i]]);
        if (ready == 0) {
          indexQueue.push(ldcscMatrix->rowIndices[i]);
        }
      }
      solved += 1;
    }
  }
  delete[] dependencies;
}

string TBBSolver::getName() {
  return "TBBSolver";
}


/**
 * CSC Solver with Open MP synchronisation primitives and STL Queue
 */
OmpStlSolver::OmpStlSolver():indexQueue() {}

void OmpStlSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  int solved;
  solved = 0;
  int *dependencies = new int[ldcscMatrix->M];
  memcpy(dependencies, initialDependencies, sizeof(int) * ldcscMatrix->M);

  #pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
    if (initialDependencies[i] == 0) {
      #pragma omp critical(index_queue_lock)
      indexQueue.push(i);
    }
  }

  #pragma omp parallel
  while (solved < ldcscMatrix->M) {

    int solve_index = -1;

    #pragma omp critical(queue)
    {
      if (!(indexQueue.empty())) {
        solve_index = indexQueue.front();
        indexQueue.pop();
      }
    }

    if (solve_index >= 0) {
      double componentSum = 0.0;
      for (
          int componentIndex = ldcsrMatrix->rowPtr[solve_index];
          componentIndex < ldcsrMatrix->rowPtr[solve_index + 1] - 1;
          componentIndex++) {
        componentSum +=
            x[ldcsrMatrix->colIndices[componentIndex]] * ldcsrMatrix->values[componentIndex];

      }
      x[solve_index] = (b[solve_index] - componentSum) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];
        int ready;
        #pragma omp atomic capture seq_cst
        ready = --(dependencies[ldcscMatrix->rowIndices[i]]);
        if (ready == 0) {
          #pragma omp critical(index_queue_lock)
          indexQueue.push(ldcscMatrix->rowIndices[i]);
        }
      }
      #pragma omp atomic update seq_cst
      solved += 1;
    }
  }
  delete[] dependencies;
}

string OmpStlSolver::getName() {
  return "OmpStlSolver";
}

/**
 * CSC Solver with Cameron lock-free queue
 */
CameronSolver::CameronSolver():indexQueue() {}

void CameronSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  int solved;
  solved = 0;
  int *dependencies = new int[ldcscMatrix->M];
  memcpy(dependencies, initialDependencies, sizeof(int) * ldcscMatrix->M);

  #pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
    if (initialDependencies[i] == 0) {
      indexQueue.enqueue(i);
    }
  }

  #pragma omp parallel
  while (solved < ldcscMatrix->M) {

    int solve_index = -1;

    if (indexQueue.try_dequeue(solve_index)) {
      double componentSum = 0.0;
      for (
          int componentIndex = ldcsrMatrix->rowPtr[solve_index];
          componentIndex < ldcsrMatrix->rowPtr[solve_index + 1] - 1;
          componentIndex++) {
        componentSum +=
            x[ldcsrMatrix->colIndices[componentIndex]] * ldcsrMatrix->values[componentIndex];

      }
      x[solve_index] = (b[solve_index] - componentSum) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];
        int ready;
        #pragma omp atomic capture seq_cst
        ready = --(dependencies[ldcscMatrix->rowIndices[i]]);
        if (ready == 0) {
          indexQueue.enqueue(ldcscMatrix->rowIndices[i]);
        }
      }
      #pragma omp atomic update seq_cst
      solved += 1;
    }
  }
  delete[] dependencies;
}

string CameronSolver::getName() {
  return "CameronSolver";
}


/**
 * CSC Solver with sequential run
 */
SeqParSolver::SeqParSolver():indexQueue() {}

void SeqParSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  int solved;
  solved = 0;
  int *dependencies = new int[ldcscMatrix->M];
  memcpy(dependencies, initialDependencies, sizeof(int) * ldcscMatrix->M);

  for (int i = 0; i < ldcscMatrix->M; i++) {
    if (initialDependencies[i] == 0) {
      indexQueue.push(i);
    }
  }

  while (solved < ldcscMatrix->M) {

    int solve_index = -1;

    if (!(indexQueue.empty())) {
      solve_index = indexQueue.front();
      indexQueue.pop();
    }

    if (solve_index >= 0) {
      double componentSum = 0.0;
      for (
          int componentIndex = ldcsrMatrix->rowPtr[solve_index];
          componentIndex < ldcsrMatrix->rowPtr[solve_index + 1] - 1;
          componentIndex++) {
        componentSum +=
            x[ldcsrMatrix->colIndices[componentIndex]] * ldcsrMatrix->values[componentIndex];

      }
      x[solve_index] = (b[solve_index] - componentSum) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];
        int ready;
        ready = --(dependencies[ldcscMatrix->rowIndices[i]]);
        if (ready == 0) {
          indexQueue.push(ldcscMatrix->rowIndices[i]);
        }
      }
      solved += 1;
    }
  }
  delete[] dependencies;
}

string SeqParSolver::getName() {
  return "SeqParSolver";
}
