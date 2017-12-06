#ifndef _METHOD_H_
#define _METHOD_H_

#include "matrix.h"
#include <string>
#include <tbb/concurrent_queue.h>
#include <queue>
#include <deque>
#include <atomic>
#include "concurrentqueue.h"
#ifdef MKL_EXISTS
#include <mkl.h>
#endif
#include <boost/lockfree/queue.hpp>

namespace thundercat {
  class SparseTriangularSolver {
  public:
    virtual void initThreads(int numThreads);
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters) = 0;

    // Solve for x in Ax=b where A is a lower triangular matrix
    // with a full diagonal. The matrix A should be set beforehand using the init method.
    // This is because some methods prefer CSR format while others like CSC.
    virtual void forwardSolve(double* __restrict b, double* __restrict x) = 0;

    virtual void backwardSolve(double* __restrict b, double* __restrict x) = 0;

    virtual std::string getName() = 0;
  };
  
  class SequentialCSRSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();
    
  private:
    CSRMatrix *ldcsrMatrix;
    CSRMatrix *udcsrMatrix;
  };

  class SequentialCSCSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();

  private:
    CSCMatrix *ldcscMatrix;
    CSCMatrix *udcscMatrix;
  };

  class ParallelCSCSolver: public SparseTriangularSolver {
  public:
      virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                        CSRMatrix* udcsr, CSCMatrix* udcsc,
                        int iters);

      virtual void backwardSolve(double* __restrict b, double* __restrict x);

  protected:
      CSCMatrix *ldcscMatrix;
      CSRMatrix *ldcsrMatrix;
      CSCMatrix *udcscMatrix;

      int* initialDependencies;

  };

  class BoostSolver: public ParallelCSCSolver {
  public:
    BoostSolver();
    
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
		      CSRMatrix* udcsr, CSCMatrix* udcsc,
		      int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();
    
    boost::lockfree::queue<int> *indexQueue;
  };
  
    class TBBSolver: public ParallelCSCSolver {
    public:
        TBBSolver();

        virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                          CSRMatrix* udcsr, CSCMatrix* udcsc,
                          int iters);

	virtual void forwardSolve(double* __restrict b, double* __restrict x);

        virtual std::string getName();

        tbb::concurrent_bounded_queue<int> indexQueue;
    };

    class OmpStlSolver: public ParallelCSCSolver {
    public:
        OmpStlSolver();

        virtual void forwardSolve(double* __restrict b, double* __restrict x);

        virtual std::string getName();

        std::queue<int> indexQueue;
  };

    class CameronSolver: public ParallelCSCSolver {
    public:
        CameronSolver();

        virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                          CSRMatrix* udcsr, CSCMatrix* udcsc,
                          int iters);
        virtual void forwardSolve(double* __restrict b, double* __restrict x);

        virtual std::string getName();

        moodycamel::ConcurrentQueue<int> indexQueue;
	tbb::atomic<int> *dependencies;
    };

    class SeqParSolver: public ParallelCSCSolver {
    public:
        SeqParSolver();

        virtual void forwardSolve(double* __restrict b, double* __restrict x);

        virtual std::string getName();

       std::queue<int> indexQueue;
    };

  class ExperimentalSolver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);

    virtual std::string getName();

  private:
    CSRMatrix *ldcsrMatrix;
    CSCMatrix *ldcscMatrix;
    CSRMatrix *udcsrMatrix;
    CSCMatrix *udcscMatrix;
    int *unknownVars;
    std::atomic<int> *dependencies;
    std::vector<std::deque<int> > rowsToSolve;
  };

  class EuroPar16Solver: public SparseTriangularSolver {
  public:
    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
    
    virtual std::string getName();
    
  private:
    CSCMatrix *ldcscMatrix;
    CSCMatrix *udcscMatrix;
    int *ldrowLengths;
    int *udrowLengths;
  };

  class MKLSolver: public SparseTriangularSolver {
  public:
    virtual void initThreads(int numThreads);

    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);
    
  protected:
    CSRMatrix *ldcsrMatrix;
    CSCMatrix *ldcscMatrix;
    CSRMatrix *udcsrMatrix;
    CSCMatrix *udcscMatrix;
  };

  class MKLCSRSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);

    virtual void backwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLCSCSolver: public MKLSolver {
  public:
    virtual std::string getName();
    
    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
  };

  class MKLInspectorExecutorSolver: public SparseTriangularSolver {
  public:
    virtual void initThreads(int numThreads);

    virtual void init(CSRMatrix* ldcsr, CSCMatrix* ldcsc,
                      CSRMatrix* udcsr, CSCMatrix* udcsc,
                      int iters);

    virtual void forwardSolve(double* __restrict b, double* __restrict x);
    
    virtual void backwardSolve(double* __restrict b, double* __restrict x);
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc) = 0;
    
  protected:
    sparse_matrix_t mklL;
    matrix_descr descL;
    sparse_matrix_t mklU;
    matrix_descr descU;
#endif
  };
  
  class MKLInspectorExecutorCSRSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc);
#endif
  };

  // NOTE: This method fails at the mkl_sparse_set_sv_hint step
  // by giving error code 6. We still keep it for completeness.
  class MKLInspectorExecutorCSCSolver: public MKLInspectorExecutorSolver {
  public:
    virtual std::string getName();
    
#ifdef MKL_EXISTS
  protected:
    virtual void createMKLMatrices(CSRMatrix *ldcsr, CSCMatrix *ldcsc,
                                   CSRMatrix *udcsr, CSCMatrix *udcsc);
#endif
  };
}

#endif
