#include "profiler.h"
#include "docopt.h"
#include "matrix.h"
#include "method.h"
#include <cmath>
#include <iostream>
#include <chrono>
#ifdef OPENMP_EXISTS
#include "omp.h"
#endif

using namespace thundercat;
using namespace std;

string filename;
CSCMatrix *ldCSCMatrix;
CSRMatrix *ldCSRMatrix;
SparseTriangularSolver *method;
double *bVector;
double *xVector;
double *xVectorReference;

bool DEBUG_MODE_ON;
int NUM_THREADS;
int ITERS;

//----------------------------------------------------------
void parseCommandLineOptions(int argc, const char *argv[]);
void initializeThreads();
void loadMatrix();
void populateVectors();
void benchmark();

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
    sptrsv <mtxFile> (sequentialCSR | sequentialCSC | mklCSR | mklCSC | mklIECSR | mklIECSC) [--threads=<num>] [--debug] [--iters=<count>]
    sptrsv (-h | --help)
    sptrsv --version

  Options:
    -h --help                     Show this screen.
    --version                     Show version.
    -d --debug                    Turn debug mode on.
    -t=<num>, --threads=<num>     Number of threads to use [default: 1].
    --iters=<count>               Number of iterations for benchmarking.
)";

void parseCommandLineOptions(int argc, const char *argv[]) {
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, { argv + 1, argv + argc }, true, "SpTRSV 0.1");
  
  DEBUG_MODE_ON = args["--debug"].asBool();

  if (args["sequentialCSR"].asBool()) {
    method = new SequentialCSRSolver;
  } else if (args["sequentialCSC"].asBool()) {
    method = new SequentialCSCSolver;
  } else if (args["mklCSR"].asBool()) {
    method = new MKLCSRSolver;
  } else if (args["mklCSC"].asBool()) {
    method = new MKLCSCSolver;
  } else if (args["mklIECSR"].asBool()) {
    method = new MKLInspectorExecutorCSRSolver;
  } else if (args["mklIECSC"].asBool()) {
    method = new MKLInspectorExecutorCSCSolver;
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
#ifdef OPENMP_EXISTS
  omp_set_num_threads(NUM_THREADS);
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

void loadMatrix() {
  MMMatrix *mmMatrix = MMMatrix::fromFile(filename);
  if (!mmMatrix->isSquare()) {
    cerr << "Only square matrices are accepted.\n";
    exit(1);
  }
  if (!mmMatrix->hasFullDiagonal()) {
    cerr << "Input matrix has to have a full diagonal.\n";
    exit(1);
  }
  
  MMMatrix *ldMatrix = mmMatrix->getLD();
  delete mmMatrix;
  ldCSCMatrix = ldMatrix->toCSC();
  ldCSRMatrix = ldMatrix->toCSR();
  delete ldMatrix;
}

void populateVectors() {
  int N = ldCSCMatrix->N;
  int M = ldCSCMatrix->M;
  xVector = new double[M];
  xVectorReference = new double[M];
  bVector = new double[N];
  
  // Initialize the input vector
  for (int j = 0; j < M; j++) {
    xVectorReference[j] = M + 2 - j;
  }
  
  // Multiply x with the matrix to obtain b.
  // This is a standard SpMV operation.
  int *rowPtr = ldCSRMatrix->rowPtr;
  int *cols = ldCSRMatrix->colIndices;
  double *vals = ldCSRMatrix->values;
  
  for (int i = 0; i < N; i++) {
    double sum = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
      sum += vals[k] * xVectorReference[cols[k]];
    }
    bVector[i] = sum;
  }
}

void validateResult() {
  int M = ldCSCMatrix->M;
  
  for (int j = 0; j < M; j++) {
    double diff = xVectorReference[j] - xVector[j];
    if (abs(diff) > 0.00001) {
      cerr << "OOPS!! Vectors different at index "
           << j << ": " << xVectorReference[j] << " vs " << xVector[j] << "\n";
    }
  }
}

int findNumIterations() {
  if (ITERS > 0) {
    // Iterations specified as a command line argument
    return ITERS;
  }
  // Find iteration count so that total execution with
  // the reference implementation will be about 4 secs.
  SequentialCSRSolver solver;
  solver.init(ldCSRMatrix, ldCSCMatrix, NUM_THREADS, 5);

  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < 5; i++) {
    solver.forwardSolve(bVector, xVector);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  long long targetDuration = 4000000;
  int iters = targetDuration * 5 / duration;
  int roundedIters = ((iters / 10) + 1) * 10;
  return roundedIters;
}

void benchmark() {
  int iters = findNumIterations();
  method->init(ldCSRMatrix, ldCSCMatrix, NUM_THREADS, iters);

  method->forwardSolve(bVector, xVector);
  if (DEBUG_MODE_ON) {
    validateResult();
  }
  
  Profiler::recordTime("SpTRSV", iters, [iters]() {
    for (unsigned i = 0; i < iters; i++) {
      method->forwardSolve(bVector, xVector);
    }
  });
  Profiler::print();
  
  if (DEBUG_MODE_ON) {
    cout << "ITERS: " << iters << "\n";
  }
}
