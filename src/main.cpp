#include "profiler.h"
#include "docopt/docopt.h"
#include "matrix.h"
#include <cmath>
#include <iostream>

using namespace thundercat;
using namespace std;

enum SpTRSVMethod {
  REFERENCE,
  MKL
};

string filename;
CSCMatrix *ldCSCMatrix;
CSRMatrix *ldCSRMatrix;
SpTRSVMethod method;
double *bVector;
double *xVector;
double *xVectorReference;

bool DEBUG_MODE_ON;
int NUM_THREADS;
int ITERS;

//----------------------------------------------------------
void parseCommandLineOptions(int argc, const char *argv[]);
void loadMatrix();
void populateVectors();
void solve();
void validateResult();

int main(int argc, const char *argv[]) {
  parseCommandLineOptions(argc, argv);
  loadMatrix();
  populateVectors();
  solve();
  validateResult();
  
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
    method = REFERENCE;
  } else if (args["mkl"]) {
    method = MKL;
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

void solve() {
  // TODO
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
