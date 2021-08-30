#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

# out=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/out_microbench/
out=/data/scratch/danielbd/out_micro/
mkdir -p "$out"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

declare -a kinds=("DENSE" "SPARSE" "RLE")

for ru in {1..400}
# for ru in {1..1}
do
    RUNUP=$(( (ru * 10) ))

    for kind in "${kinds[@]}"
    do
        for i in {0..9}
        # for i in {0..0}
        do
            BENCH=micro_constmul BUILD_DIR=$TMP_BUILD_DIR BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
            BENCH=micro_elemwise BUILD_DIR=$TMP_BUILD_DIR BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
        done
    done
done
