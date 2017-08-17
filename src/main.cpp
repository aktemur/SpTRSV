#include "profiler.h"
#include "docopt/docopt.h"
#include "matrix.h"
#include "method.h"
#include <cmath>
#include <iostream>

using namespace thundercat;
using namespace std;

string filename;
CSCMatrix *ldCSCMatrix;
CSRMatrix *ldCSRMatrix;
SpTRSVMethod *method;
double *bVector;
double *xVector;
double *xVectorReference;

bool DEBUG_MODE_ON;
int NUM_THREADS;

//----------------------------------------------------------
void parseCommandLineOptions(int argc, const char *argv[]);
void loadMatrix();
void populateVectors();
void benchmark();

int main(int argc, const char *argv[]) {
  parseCommandLineOptions(argc, argv);
  loadMatrix();
  populateVectors();
  benchmark();
  
  return 0;
}

static const char USAGE[] =
R"(OzU SRL SpTRSV.

  Usage:
    sptrsv <mtxFile> (reference | mkl) [--threads=<num>] [--debug]
    sptrsv (-h | --help)
    sptrsv --version

  Options:
    -h --help            Show this screen.
    --version            Show version.
    -d --debug           Turn debug mode on
    --threads=<num>      Number of threads to use [default: 1].
)";

void parseCommandLineOptions(int argc, const char *argv[]) {
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, { argv + 1, argv + argc }, true, "SpTRSV 0.1");

  if (args["reference"]) {
    method = new ReferenceSolver;
  } else if (args["mkl"]) {
    method = new MKLSolver;
  } else {
    cerr << "Unexpection situation occurred while parsing the method.\n";
    exit(1);
  }
  
  NUM_THREADS = args["--threads"].asLong();
  if (NUM_THREADS <= 0) {
    cerr << "Number of threads has to be positive.\n";
    exit(1);
  }

  DEBUG_MODE_ON = args["--debug"].asBool();

  filename = args["<mtxFile>"].asString();
}

void loadMatrix() {
  MMMatrix *mmMatrix = MMMatrix::fromFile(filename);
  if (mmMatrix->N != mmMatrix->M) {
    cerr << "Only square matrices are accepted.\n";
    exit(1);
  }
  
  MMMatrix *ldMatrix = mmMatrix->getLD();
  delete mmMatrix;
  ldCSCMatrix = ldMatrix->toCSC();
  ldCSRMatrix = ldMatrix->toCSR();
  delete ldMatrix;
  
  // Validate the matrix
  // The diagonal has to be full.
  for (int j = 0; j < ldCSCMatrix->M; j++) {
    int k = ldCSCMatrix->colPtr[j];
    // The first element of the column has to be nonzero
    // and it has to be at row index j
    if (ldCSCMatrix->rowIndices[k] != j || ldCSCMatrix->values[k] == 0) {
      cerr << "The given matrix does not have a full diagonal.\n";
      exit(1);
    }
  }
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

void solve() {
  method->solve(ldCSCMatrix, bVector, xVector);
}

void validateResult() {
  int M = ldCSCMatrix->M;
  
  for (int j = 0; j < M; j++) {
    double diff = xVectorReference[j] - xVector[j];
    if (abs(diff) > 0.000001) {
      cerr << "OOPS!! Vectors different at index "
           << j << ": " << xVectorReference[j] << " vs " << xVector[j] << "\n";
    }
  }
}

int findNumIterations() {
  // Find iteration count so that total execution will be about 2 secs.
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < 5; i++) {
    solve();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  long long targetDuration = 2000000;
  int iters = targetDuration * 5 / duration;
  int roundedIters = ((iters / 10) + 1) * 10;
  return roundedIters;
}

void benchmark() {
  solve();
  validateResult();
  
  int iters = findNumIterations();
  cout << "ITERS: " << iters << "\n";
  
  Profiler::recordTime("SpTRSV", iters, [iters]() {
    for (unsigned i = 0; i < iters; i++) {
      solve();
    }
  });

  Profiler::print();
}
