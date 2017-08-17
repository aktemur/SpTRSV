#include "profiler.h"
#include "docopt/docopt.h"
#include <iostream>
#include "matrix.h"

using namespace thundercat;
using namespace std;

static const char USAGE[] =
R"(OzU SRL SpTRSV.

    Usage:
      sptrsv <mtxFile> (Reference | MKL) [--threads=<num>] [--debug]
      sptrsv (-h | --help)
      sptrsv --version

    Options:
      -h --help            Show this screen.
      --version            Show version.
      -d --debug           Turn debug mode on
      --threads=<num>      Number of threads to use [default: 1].
)";


MMMatrix *mmMatrix;
bool DEBUG_MODE_ON;
int NUM_THREADS;
int ITERS;

void parseCommandLineOptions(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
  parseCommandLineOptions(argc, argv);

  mmMatrix->print();
  cout << "Debug? : " << DEBUG_MODE_ON << "\n";
  cout << "Threads: " << NUM_THREADS << "\n";
  
  return 0;
}

void parseCommandLineOptions(int argc, const char *argv[]) {
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE,
                     { argv + 1, argv + argc },
                     true,               // show help if requested
                     "SpTRSV 0.1");  // version string

  mmMatrix = MMMatrix::fromFile(args["<mtxFile>"].asString());

  NUM_THREADS = args["--threads"].asLong();
  if (NUM_THREADS <= 0) {
    cerr << "Number of threads has to be positive.\n";
    exit(1);
  }

  DEBUG_MODE_ON = args["--debug"].asBool();
}
