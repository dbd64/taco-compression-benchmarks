#!/bin/bash
set -u

RU_UPPER_BOUND=40
NUM_RAND=10
RUN_MULTIPLIER=10
REPETITIONS=100
RAND_WIDTH=1000
RAND_HEIGHT=10000

while getopts s flag
do
    case "${flag}" in
        s) RU_UPPER_BOUND=10
           NUM_RAND=3
           RUN_MULTIPLIER=40
           REPETITIONS=10
           ;;
    esac
done

SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/micro
lanka=OFF
mkdir -p "$out/constmul/"
mkdir -p "$out/elemwise/"
mkdir -p "$out/maskmul/"
mkdir -p "$out/spmv/"

make taco/build/taco-bench

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
for ru in $( seq 1 $RU_UPPER_BOUND ) 
do
    RUNUP=$(( (ru * $RUN_MULTIPLIER) ))

    for kind in "${kinds[@]}"
    do
        for i in $( seq 0 $(( ($NUM_RAND - 1) )) )
        do
            BENCH=micro_constmul RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/constmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
            BENCH=micro_elemwise RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/elemwise/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep          
            BENCH=micro_maskmul RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/maskmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
            BENCH=spmv_rand RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/spmv/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
        done
    done
done

python3 $SCRIPT_DIR/merge_csv.py $out/constmul $out/constmul.csv
python3 $SCRIPT_DIR/merge_csv.py $out/elemwise $out/elemwise.csv
python3 $SCRIPT_DIR/merge_csv.py $out/maskmul $out/maskmul.csv
python3 $SCRIPT_DIR/merge_csv.py $out/spmv $out/spmv.csv
rm -r "$out/constmul/"
rm -r "$out/elemwise/"
rm -r "$out/maskmul/"
rm -r "$out/spmv/"
