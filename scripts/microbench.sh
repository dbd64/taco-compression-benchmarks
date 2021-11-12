#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/micro/
lanka=OFF
mkdir -p "$out"

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
for ru in {1..600}
do
    RUNUP=$(( (ru * 10) ))

    for kind in "${kinds[@]}"
    do
        for i in {0..9}
        do
            BENCH=micro_constmul BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
            BENCH=micro_elemwise BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench          
            BENCH=micro_maskmul BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
            BENCH=spmv_rand BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
        done
    done
done
