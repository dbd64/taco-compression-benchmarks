#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/micro
lanka=OFF
mkdir -p "$out/constmul/"
mkdir -p "$out/elemwise/"
mkdir -p "$out/maskmul/"
mkdir -p "$out/spmv/"

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
for ru in {1..600}
do
    RUNUP=$(( (ru * 10) ))

    for kind in "${kinds[@]}"
    do
        for i in {0..9}
        do
            BENCH=micro_constmul BENCH_KIND=$kind OUTPUT_PATH=$out/constmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
            BENCH=micro_elemwise BENCH_KIND=$kind OUTPUT_PATH=$out/elemwise/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench          
            BENCH=micro_maskmul BENCH_KIND=$kind OUTPUT_PATH=$out/maskmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
            BENCH=spmv_rand BENCH_KIND=$kind OUTPUT_PATH=$out/spmv/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench
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
