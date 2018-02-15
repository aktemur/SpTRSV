#include "profiler.h"
#include "docopt.h"
#include "solver.hpp"
#include "sequential.hpp"
#include "mklsolver.hpp"
#include "europar16.hpp"
#include "experimental.hpp"
#include "hpcs17.hpp"
#include "mmmatrix.hpp"
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

#define VALUETYPE double

using namespace thundercat;
using namespace std;

string filename;
unique_ptr<CSCMatrix<VALUETYPE>> ldCSCMatrix;
unique_ptr<CSRMatrix<VALUETYPE>> ldCSRMatrix;
unique_ptr<CSCMatrix<VALUETYPE>> udCSCMatrix;
unique_ptr<CSRMatrix<VALUETYPE>> udCSRMatrix;
unique_ptr<SparseTriangularSolver<VALUETYPE>> method;
VALUETYPE *bVector;
VALUETYPE *xVector;
VALUETYPE *yVector;
VALUETYPE *xVectorReference;
VALUETYPE *yVectorReference;

bool DEBUG_MODE_ON;
int NUM_THREADS;
int ITERS;

// allocate a large buffer to flush out cache
static const size_t LLC_CAPACITY = 32*1024*1024;
static const VALUETYPE *bufToFlushLlc = NULL;


//----------------------------------------------------------
void parseCommandLineOptions(int argc, const char *argv[]);
void initializeThreads();
void loadMatrix();
void populateVectors();
void benchmark();
void flushLLC();

int main(int argc, const char *argv[]) {
  parseCommandLineOptions(argc, argv);
  initializeThreads();
  loadMatrix();
  populateVectors();
  benchmark();
  
  return 0;
}

static const char USAGE[] =
R"(OzU SRL SpTRSV.

  Usage:
    sptrsv <mtxFile> (seqCSR | seqCSC | seqCSRLen | mklCSR | mklCSC | mklIECSR | europar16 | hpcs17 | experimental)
      [--threads=<num>] [--debug] [--iters=<count>]
    sptrsv (-h | --help)
    sptrsv --version

  Options:
    -h --help                     Show this screen.
    --version                     Show version.
    -d --debug                    Turn debug mode on.
    --threads=<num>               Number of threads to use [default: 1].
    --iters=<count>               Number of iterations for benchmarking.
)";

void parseCommandLineOptions(int argc, const char *argv[]) {
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, { argv + 1, argv + argc }, true, "SpTRSV 0.1");
  
  DEBUG_MODE_ON = args["--debug"].asBool();

  if (args["seqCSR"].asBool()) {
    method = make_unique<SequentialCSRSolver<VALUETYPE>>();
  } else if (args["seqCSC"].asBool()) {
    method = make_unique<SequentialCSCSolver<VALUETYPE>>();
  } else if (args["seqCSRLen"].asBool()) {
    method = make_unique<SequentialCSRLenSolver<VALUETYPE>>();
  } else if (args["mklCSR"].asBool()) {
    method = make_unique<MKLCSRSolver<VALUETYPE>>();
  } else if (args["mklCSC"].asBool()) {
    method = make_unique<MKLCSCSolver<VALUETYPE>>();
  } else if (args["mklIECSR"].asBool()) {
    method = make_unique<MKLInspectorExecutorCSRSolver<VALUETYPE>>();
  } else if (args["europar16"].asBool()) {
    method = make_unique<EuroPar16Solver<VALUETYPE>>();
  } else if (args["hpcs17"].asBool()) {
    method = make_unique<HPCS17Solver<VALUETYPE>>();
  } else if (args["experimental"].asBool()) {
    method = make_unique<ExperimentalSolver<VALUETYPE>>();
  } else {
    cerr << "Unexpection situation occurred while parsing the method.\n";
    exit(1);
  }
  if (DEBUG_MODE_ON) {
    cout << "Method: " << method->getName() << "\n";
  }
  
  NUM_THREADS = args["--threads"].asLong();
  if (NUM_THREADS <= 0) {
    cerr << "Number of threads has to be positive.\n";
    exit(1);
  }

  if (args["--iters"]) {
    ITERS = args["--iters"].asLong();
  } else {
    ITERS = -1;
  }
  
  filename = args["<mtxFile>"].asString();
}

void initializeThreads() {
  method->initThreads(NUM_THREADS);
}

void loadMatrix() {
  unique_ptr<MMMatrix<VALUETYPE>> mmMatrix = MMMatrix<VALUETYPE>::fromFile(filename);
  if (!mmMatrix->isSquare()) {
    cerr << "Only square matrices are accepted.\n";
    exit(1);
  }
  if (!mmMatrix->hasFullDiagonal()) {
    cerr << "Input matrix has to have a full diagonal.\n";
    exit(1);
  }
  
  unique_ptr<MMMatrix<VALUETYPE>> ldMatrix = mmMatrix->getLD();
  unique_ptr<MMMatrix<VALUETYPE>> udMatrix = mmMatrix->getUD();
  mmMatrix.release();
  ldCSCMatrix = ldMatrix->toCSC();
  ldCSRMatrix = ldMatrix->toCSR();
  udCSCMatrix = udMatrix->toCSC();
  udCSRMatrix = udMatrix->toCSR();
}

void spmvCSR(unique_ptr<CSRMatrix<VALUETYPE>> const &csr, const VALUETYPE *v, VALUETYPE *w) {
  int N = csr->N;
  int *rowPtr = csr->rowPtr;
  int *cols = csr->colIndices;
  VALUETYPE *vals = csr->values;
  
  for (int i = 0; i < N; i++) {
    VALUETYPE sum = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
      sum += vals[k] * v[cols[k]];
    }
    w[i] = sum;
  }
}

void populateVectors() {
  int N = ldCSRMatrix->N;
  xVector = new VALUETYPE[N];
  yVector = new VALUETYPE[N];
  xVectorReference = new VALUETYPE[N];
  yVectorReference = new VALUETYPE[N];
  bVector = new VALUETYPE[N];
  
  // Initialize the input vector
  for (int j = 0; j < N; j++) {
    xVectorReference[j] = N + 2 - j;
  }
  
  // Perform the following multiplications to obtain vector b.
  // Ux = y and Ly = b (i.e. L(Ux) = b)
  // These are two standard SpMV operations.
  spmvCSR(udCSRMatrix, xVectorReference, yVectorReference);
  spmvCSR(ldCSRMatrix, yVectorReference, bVector);
}

void validateResult() {
  int N = ldCSRMatrix->N;
  
  for (int j = 0; j < N; j++) {
    VALUETYPE diffx = xVectorReference[j] - xVector[j];
    if (fabs(diffx)/fabs(xVectorReference[j]) > 0.00001) {
      cerr << "OOPS!! xVectors different at index "
           << j << ": " << xVectorReference[j] << " vs " << xVector[j] << "\n";
    }
    VALUETYPE diffy = yVectorReference[j] - yVector[j];
    if (fabs(diffy)/fabs(yVectorReference[j]) > 0.00001) {
      cerr << "OOPS!! yVectors different at index "
      << j << ": " << yVectorReference[j] << " vs " << yVector[j] << "\n";
    }
  }
}

int findNumIterations() {
  if (ITERS > 0) {
    // Iterations specified as a command line argument
    return ITERS;
  }
  return 128;
}

void benchmark() {
  if (posix_memalign((void**)&bufToFlushLlc, 64, LLC_CAPACITY) != 0) {
    cerr << "Problem occured in posix_memalign.\n";
    exit(1);
  }

  int iters = findNumIterations();
  if (DEBUG_MODE_ON) {
    cout << "ITERS: " << iters << "\n";
  }
  method->init(ldCSRMatrix.get(), ldCSCMatrix.get(), udCSRMatrix.get(), udCSCMatrix.get(), iters);

  method->forwardSolve(bVector, yVector);
  method->backwardSolve(yVector, xVector);
  if (DEBUG_MODE_ON) {
    validateResult();
  }

  long long *forwardDurations = new long long[iters];
  long long *backwardDurations = new long long[iters];
  
  for (unsigned i = 0; i < iters; i++) {
    flushLLC();
    auto t_0 = std::chrono::high_resolution_clock::now();
    method->forwardSolve(bVector, yVector);
    auto t_1 = std::chrono::high_resolution_clock::now();
    method->backwardSolve(yVector, xVector);
    auto t_2 = std::chrono::high_resolution_clock::now();

    auto delta_1 = std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
    forwardDurations[i] = delta_1;
    auto delta_2 = std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
    backwardDurations[i] = delta_2;
  }

  sort(forwardDurations, forwardDurations + iters);
  long long forwardDuration = forwardDurations[iters/2];
  printf("Fwd_solve: %lld ", forwardDuration);
  sort(backwardDurations, backwardDurations + iters);
  long long backwardDuration = backwardDurations[iters/2];
  printf("Bwd_solve: %lld\n", backwardDuration);
}

void flushLLC()
{
  VALUETYPE sum = 0;
#pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < LLC_CAPACITY/sizeof(bufToFlushLlc[0]); ++i) {
    sum += bufToFlushLlc[i];
  }
  FILE *fp = fopen("/dev/null", "w");
  fprintf(fp, "%f\n", sum);
  fclose(fp);
}
