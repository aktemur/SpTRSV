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
    IFS=' ' read -r -a info <<< "$line"
    groupName=${info[0]}
    matrixName=${info[1]}

    echo -n "$groupName"/"$matrixName "

    cd ..
    folderName=data/"$groupName"/"$matrixName"/"$methodName""$param1""$param2""$param3""$param4"
    mkdir -p "$folderName"
    rm -f "$folderName"/runtime.txt

    ./build/sptrsv $MATRICES/$groupName/$matrixName/$matrixName".mtx" $methodName $param1 $param2 $param3 $param4 | grep "SpTRSV" | awk '{print $2}' >> "$folderName"/runtime.txt
    ./build/sptrsv $MATRICES/$groupName/$matrixName/$matrixName".mtx" $methodName $param1 $param2 $param3 $param4 | grep "SpTRSV" | awk '{print $2}' >> "$folderName"/runtime.txt
    ./build/sptrsv $MATRICES/$groupName/$matrixName/$matrixName".mtx" $methodName $param1 $param2 $param3 $param4 | grep "SpTRSV" | awk '{print $2}' >> "$folderName"/runtime.txt
    cd tools

done < matrixNames.txt

echo " "

findMins() {
    currentTime=`date +%Y.%m.%d_%H.%M`
    local fileName=../data/$HOSTNAME.sptrsv."$methodName""$param1""$param2""$param3""$param4".$currentTime.csv
    rm -f $fileName
    while read line
    do
        IFS=' ' read -r -a info <<< "$line"
        groupName=${info[0]}
        matrixName=${info[1]}

        cd ../data/"$groupName"/"$matrixName"/"$methodName""$param1""$param2""$param3""$param4"
	    runTimes=`cat runtime.txt`
	    cd - > /dev/null
	    echo -n $groupName" "$matrixName" "  >> $fileName
	    ./findMinTiming.py $runTimes >> $fileName
	    echo     ""  >> $fileName
    done < matrixNames.txt
}

findMins

echo "Subject: $methodName SpTRSV on $HOSTNAME 

$methodName $param1 $param2 $param3 $param4 test on $HOSTNAME has finished.
" | ssmtp baris.aktemur@ozyegin.edu.tr
