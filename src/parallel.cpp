#include "method.h"
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>

using namespace thundercat;
using namespace std;


ParallelCSCSolver::ParallelCSCSolver() : indexQueue() {

}

void ParallelCSCSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                             CSRMatrix *udcsr, CSCMatrix *udcsc, int numThreads, int iters) {
  ldcscMatrix = ldcsc;
  ldcsrMatrix = ldcsr;
  udcscMatrix = udcsc;
}

void ParallelCSCSolver::forwardSolve(double *__restrict b, double *__restrict x) {

  int solved = 0;
  double *leftsum = new double[ldcscMatrix->M];
  int *dependencies = new int[ldcscMatrix->M];


//#pragma omp parallel for
  for (int i = 0; i < ldcscMatrix->M; i++) {
    leftsum[i] = 0.0;
    dependencies[i] = (ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i]) - 1;

    if (dependencies[i] == 0) {
//#pragma omp critical(indexQueue)
      indexQueue.push(i);
    }
  }

//#pragma omp parallel
  while (solved < ldcscMatrix->M) {
    int solve_index;

    while (indexQueue.empty() && solved < ldcscMatrix->M);
//#pragma omp critical(indexQueue)
    {
      if (!(indexQueue.empty())) {
        solve_index = indexQueue.front();
        indexQueue.pop();
      }
    }
    x[solve_index] =
        (b[solve_index] - leftsum[solve_index]) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];

    for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
      int row = ldcscMatrix->rowIndices[i];
      int ready;
//#pragma omp atomic update seq_cst
      leftsum[row] += ldcscMatrix->values[i] * x[solve_index];
//#pragma omp atomic capture seq_cst
      ready = --dependencies[ldcscMatrix->rowIndices[i]];
      if (ready == 0) {
//#pragma omp critical(indexQueue)
        indexQueue.push(ldcscMatrix->rowIndices[i]);
      }
    }
//#pragma omp atomic update seq_cst
    solved += 1;
  }


  delete[] leftsum;
  delete[] dependencies;
}

//void ParallelCSCSolver::forwardSolve(double *__restrict b, double *__restrict x) {
//  int solved = 0 ;
//  long global_solve_index = 0;
//  double *leftsum = new double[ldcscMatrix->M];
//  int *dependencies = new int[ldcscMatrix->M];
//
////#pragma omp critical
////  cout << "GI0: " << global_solve_index << " ";
//#pragma omp parallel for shared(leftsum, dependencies)
//  for (int i = 0; i < ldcscMatrix->M; i++) {
//    leftsum[i] =  0.0;
//    dependencies[i] =  (ldcsrMatrix->rowPtr[i+1] - ldcsrMatrix->rowPtr[i]) - 1;
//  }
//  long solve_index;
//
//#pragma omp parallel shared(global_solve_index, leftsum, solved, dependencies) private(solve_index)
//  while (solved < ldcscMatrix->M) {
//
//#pragma omp atomic capture
//    solve_index = global_solve_index++;
//    solve_index %= ldcscMatrix->M;
//
//    if (dependencies[solve_index] == 0) {
//
//      x[solve_index] = (b[solve_index] - leftsum[solve_index]) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];
//      for (int i = ldcscMatrix->colPtr[solve_index]; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
//        int row = ldcscMatrix->rowIndices[i];
//
//#pragma omp atomic update
//        leftsum[row] += ldcscMatrix->values[i] * x[solve_index];
//#pragma omp atomic update
//        dependencies[ldcscMatrix->rowIndices[i]] -= 1;
//
//      }
//#pragma omp atomic update
//      solved += 1;
//    }
//  }
////#pragma omp critical
////  cout << "GI:" << global_solve_index << " ldcscMatrix->M:" << ldcscMatrix->M << endl;
//  delete[] leftsum;
//  delete[] dependencies;
//}

//void ParallelCSCSolver::forwardSolve(double *__restrict b, double *__restrict x) {
//  atomic<int> solved{0};
//  atomic<long> global_solve_index{-1};
//  atomic<double> *leftsum = new atomic<double>[ldcscMatrix->M];
//  atomic<int> *dependencies = new atomic<int>[ldcscMatrix->M];
//
//  #pragma omp parallel for
//  for (int i = 0; i < ldcscMatrix->M; i++) {
//    atomic_init(&(leftsum[i]), 0.0);
//    atomic_init(&(dependencies[i]), ldcsrMatrix->rowPtr[i+1] - ldcsrMatrix->rowPtr[i]);
//  }
//
//#pragma omp parallel shared(global_solve_index, leftsum, solved, dependencies)
//  while (solved < ldcscMatrix->M) {
//    long solve_index = (++global_solve_index) % ldcscMatrix->M;
//    int expected = 1;
//
//    if (dependencies[solve_index].compare_exchange_weak(expected, 0)) {
////    if (dependencies[solve_index] == expected) {
//      dependencies[solve_index] = 0;
//      x[solve_index] = (b[solve_index] - leftsum[solve_index]) / ldcscMatrix->values[ldcscMatrix->colPtr[solve_index]];
//
//      for (int i = ldcscMatrix->colPtr[solve_index] + 1; i < ldcscMatrix->colPtr[solve_index + 1]; i++) {
//        int row = ldcscMatrix->rowIndices[i];
//
//        double partialLeftSum = ldcscMatrix->values[i] * x[solve_index];
//        double expected, desired;
//        do {
//          expected = leftsum[row].load();
//          desired = expected + partialLeftSum;
//        } while (!leftsum[row].compare_exchange_weak(expected, desired));
////        leftsum[row] += ldcscMatrix->values[i] * x[solve_index];
//
//        dependencies[ldcscMatrix->rowIndices[i]] -= 1;
//      }
//      solved += 1;
//    }
//  }
//  delete[] leftsum;
//  delete[] dependencies;
//}

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
