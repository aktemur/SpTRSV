#!/usr/bin/env bash

ITERATIONS=5
THREADS=$(echo 1 2 4 8)
METHODS=$(echo tbbCSC camCSC ompCSC)
MATRICES=$(ls ../data)


for MATRIX in $MATRICES;
do
  echo -ne "$MATRIX\tseqCSC\t";
  ../build_ubuntu/./sptrsv ../data/$MATRIX seqCSC --threads 1 --iters $ITERATIONS;

    echo -ne "$MATRIX\tseqparCSC\t";
  ../build_ubuntu/./sptrsv ../data/$MATRIX seqparCSC --threads 1 --iters $ITERATIONS;
  for THREAD in $THREADS;
  do
    for METHOD in $METHODS;
    do
      echo -ne "$MATRIX\t$METHOD--$THREAD\t";
      ../build_ubuntu/./sptrsv ../data/$MATRIX $METHOD --threads $THREAD --iters $ITERATIONS;
    done;
  done;
done;