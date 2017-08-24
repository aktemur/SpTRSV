#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 '<sptrsv parameters>'" >&2
  exit 1
fi

if [ -z ${MATRICES+x} ]; then
    echo "Set MATRICES variable to the matrices folder."
    exit 1
fi

source /opt/intel/bin/compilervars.sh intel64

methodName=$1
param1=$2
param2=$3
param3=$4
param4=$5

echo "Running SpTRSV test " $methodName $param1 $param2

while read line
do
    groupName=$(dirname $line)
    matrixName=$(basename $line)

    echo ----- "$groupName"/"$matrixName"

    cd ..
    ./build/sptrsv $MATRICES/$groupName/$matrixName/$matrixName".mtx" $methodName -d --iters=1 $param1 $param2 $param3 $param4
    cd tools

done < matrixNames.txt

echo " "
