#include "method.h"
#include <iostream>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

using namespace thundercat;
using namespace std;

extern bool DEBUG_MODE_ON;

void SparseTriangularSolver::initThreads(int numThreads) {
#ifdef OPENMP_EXISTS
  omp_set_num_threads(numThreads);
  int nthreads = -1;
#pragma omp parallel
  {
#pragma omp master
    {
      nthreads = omp_get_num_threads();
    }
  }
  if (DEBUG_MODE_ON)
    cout << "NumThreads: " << nthreads << "\n";
#endif
}
