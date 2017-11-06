#include "method.h"
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>

using namespace thundercat;
using namespace std;

void ParallelCSCSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                             CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcscMatrix = ldcsc;
  udcscMatrix = udcsc;
}

void ParallelCSCSolver::forwardSolve(double *__restrict b, double *__restrict x) {
  atomic<int> solved{0};
  atomic<long> global_solve_index{-1};
  double *leftsum = new double[ldcscMatrix->M];
//  atomic<double> *leftsum = new atomic<double>[ldcscMatrix->M];

//  #pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
//    atomic_init(&(leftsum[i]), 0.0);
    leftsum[i] = 0;
  }

  atomic<int> *dependencies = new atomic<int>[ldcscMatrix->M];

  omp_set_num_threads(4);

#pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
    atomic_init(&(dependencies[i]), 0);

  }
#pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->NZ; i++) {
    dependencies[ldcscMatrix->rowIndices[i]] += 1;
  }

#pragma omp parallel shared(global_solve_index, leftsum, solved, dependencies)
  while (solved < ldcscMatrix->M) {
    long solve_index = (++global_solve_index) % ldcscMatrix->M;
    int expected = 1;

//    if (dependencies[solve_index].compare_exchange_weak(expected, 0)) {
    if (dependencies[solve_index] == expected) {
      dependencies[solve_index] = 0;
      x[solve_index] = (b[solve_index] - leftsum[solve_index]) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

      for (int i = ldcscMatrix->colPtr[solve_index] + 1; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
        int row = ldcscMatrix->rowIndices[i];

//        double partialLeftSum = ldcscMatrix->values[i] * x[solve_index];
//        double expected, desired;
//        do {
//          expected = leftsum[row].load();
//          desired = expected + partialLeftSum;
//        } while (!leftsum[row].compare_exchange_weak(expected, desired));
        leftsum[row] += ldcscMatrix->values[i] * x[solve_index];


        dependencies[ldcscMatrix->rowIndices[i]] -= 1;
      }
      solved += 1;
    }
  }
  delete[] leftsum;
  delete[] dependencies;
}

void ParallelCSCSolver::backwardSolve(double *__restrict b, double *__restrict x) {
  double *rightsum = new double[udcscMatrix->M];
  memset(rightsum, 0, sizeof(double) * udcscMatrix->M);
  for (int j = udcscMatrix->M - 1; j >= 0; j--) {
    x[j] = (b[j] - rightsum[j]) / udcscMatrix->values[udcscMatrix->colPtr[j + 1] - 1];
//#pragma omp parallel for
    for (int k = udcscMatrix->colPtr[j + 1] - 1 - 1; k >= udcscMatrix->colPtr[j]; k--) {
      int row = udcscMatrix->rowIndices[k];
      rightsum[row] += udcscMatrix->values[k] * x[j];
    }
  }

  delete[] rightsum;
}

string ParallelCSCSolver::getName() {
  return "ParallelCSC";
}
