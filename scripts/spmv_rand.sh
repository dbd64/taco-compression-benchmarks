#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

# out=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/
out=/data/scratch/danielbd/out_spmv_rand/
# out=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/out_spmv_rand/
mkdir -p "$out"

declare -a kinds=("DENSE" "SPARSE" "RLE")
# declare -a kinds=("SPARSE")

for ru in {1..1000}
# for ru in {1..1}
do
    RUNUP=$(( (ru * 10) ))

    for kind in "${kinds[@]}"
    do
        for i in {0..9}
        do
            BENCH=spmv_rand BENCH_KIND=$kind OUTPUT_PATH=$out RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
        done
    done
done
