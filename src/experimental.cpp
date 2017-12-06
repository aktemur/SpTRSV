#include "method.h"
#include <cstring>
#include <iostream>
#include <atomic>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

using namespace thundercat;
using namespace std;

#define PACKSIZE 1

ExperimentalSolver::ExperimentalSolver(): taskQueue() {
  // empty
}

void ExperimentalSolver::init(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
			      CSRMatrix *udcsr, CSCMatrix *udcsc, int iters) {
  ldcsrMatrix = ldcsr;
  ldcscMatrix = ldcsc;
  udcsrMatrix = udcsr;
  udcscMatrix = udcsc;
  const int N = ldcscMatrix->N;
  unknownVars = new int[N / PACKSIZE];
  rowsToSolve.resize(omp_get_max_threads());
  
  int *levels = new int[N / PACKSIZE];
  int maxLevel = 0;
  
  memset(unknownVars, 0, sizeof(int) * (N / PACKSIZE));
  memset(levels, 0, sizeof(int) * (N / PACKSIZE));
  
  for (int i = 0; i < N; i++) {
    const int pack = i / PACKSIZE;
    const int packBegin = pack * PACKSIZE;
    const int packEnd = packBegin + PACKSIZE;
    const int length = ldcsrMatrix->rowPtr[i + 1] - ldcsrMatrix->rowPtr[i];
    int outerDependencies = length - 1;
    for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
      const int row = ldcscMatrix->rowIndices[k];
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
  
  dependencies = new atomic<int>[N / PACKSIZE];

  dependencyGraph.resize(maxLevel + 1);
  for (int pack = 0; pack < N / PACKSIZE; pack++) {
    const int level = levels[pack];
    dependencyGraph[level].push_back(pack);
  }
  delete[] levels;
}

void ExperimentalSolver::forwardSolve(double* __restrict b, double* __restrict x) {
  const int N = ldcscMatrix->N;
  assert(sizeof(int) == sizeof(atomic<int>));
  memcpy(dependencies, unknownVars, sizeof(atomic<int>) * (N / PACKSIZE));

  levels(b, x);
}

void ExperimentalSolver::levels(double* __restrict b, double* __restrict x) {
  for (int level = 0; level < dependencyGraph.size(); level++) {
    #pragma omp parallel for
    for (int k = 0; k < dependencyGraph[level].size(); k++) {
      const int pack = dependencyGraph[level][k];
      const int packBegin = pack * PACKSIZE;
      for (int p = 0; p < PACKSIZE; p++) {
	const int i = packBegin + p;
	double leftsum = 0;
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

void ExperimentalSolver::localQueue(double* __restrict b, double* __restrict x) {
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
      //printf("Thread %d\n", threadId);
      const int pack = rowsToSolve[threadId].front();
      rowsToSolve[threadId].pop_front();
      const int packBegin = pack * PACKSIZE;
      const int packEnd = packBegin + PACKSIZE;
      for (int p = 0; p < PACKSIZE; p++) {
	const int i = packBegin + p;
	//printf("Row %d\n", i);
	double leftsum = 0;
	int j;
	for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
	  const int col = ldcsrMatrix->colIndices[j];
	  leftsum += ldcsrMatrix->values[j] * x[col];
	}
	x[i] = (b[i] - leftsum) / ldcsrMatrix->values[j];
	for (int k = ldcscMatrix->colPtr[i] + 1; k < ldcscMatrix->colPtr[i+1]; k++) {
	  const int row = ldcscMatrix->rowIndices[k];
	  const int dependentPack = row / PACKSIZE;
	  //printf("k %d %d %d\n", i, row, dependentPack);
	  if (dependentPack != pack && --(dependencies[dependentPack]) == 0) {
	    rowsToSolve[threadId].push_back(dependentPack);
	    //printf("push %d : %d\n", pack, dependentPack);
	  }
	}
      }
    }
  }
}

void ExperimentalSolver::sharedQueue(double* __restrict b, double* __restrict x) {
  atomic<int> solved = 0;
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
	const int packEnd = packBegin + PACKSIZE;
	for (int p = 0; p < PACKSIZE; p++) {
	  const int i = packBegin + p;
	  double leftsum = 0;
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

void ExperimentalSolver::spinWait(double* __restrict b, double* __restrict x) {
  const int N = ldcscMatrix->N;

  #pragma omp parallel for
  for (int pack = 0; pack < N / PACKSIZE; pack++) {
    while (dependencies[pack] > 0) {
      // spin wait
    }
    const int packBegin = pack * PACKSIZE;
    const int packEnd = packBegin + PACKSIZE;
    for (int p = 0; p < PACKSIZE; p++) {
      const int i = packBegin + p;
      double leftsum = 0;
      int j;
      for (j = ldcsrMatrix->rowPtr[i]; j < ldcsrMatrix->rowPtr[i + 1] - 1; j++) {
	int col = ldcsrMatrix->colIndices[j];
	leftsum += ldcsrMatrix->values[j] * x[col];
      }
      double xi = (b[i] - leftsum) / ldcsrMatrix->values[j];
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

void ExperimentalSolver::backwardSolve(double* __restrict b, double* __restrict x) {
  // No parallelization yet.
  for (int i = udcsrMatrix->N - 1; i >= 0; i--) {
    double sum = 0.0;
    int k;
    for (k = udcsrMatrix->rowPtr[i+1] - 1; k > udcsrMatrix->rowPtr[i]; k--) {
      int col = udcsrMatrix->colIndices[k];
      sum += udcsrMatrix->values[k] * x[col];
    }
    x[i] = (b[i] - sum) / udcsrMatrix->values[k];
  }
}

string ExperimentalSolver::getName() {
  return "Experimental";
}
