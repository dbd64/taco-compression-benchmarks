#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

REPETITIONS=100

while getopts s flag
do
    case "${flag}" in
        s) REPETITIONS=10
           ;;
    esac
done

SCRIPT_DIR=/data/scratch/danielbd/artifact/tcb/scripts
ARTIFACT_DIR=$SCRIPT_DIR/../../

echo $SCRIPT_DIR

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/spmv_rand
lanka=ON
mkdir -p "$out"
mkdir -p "$out/rawmtx"
mkdir -p "$out/hist"

if [[ ! -v LD_LIBRARY_PATH ]]; then
    LD_LIBRARY_PATH=$SCRIPT_DIR/..
fi

LANKA=$lanka make taco/build/taco-bench

declare -a kinds=("DENSE" "SPARSE" "RLE" "RLEP" "LZ77")
declare -a kinds=("DENSE" "LZ77" "RLEP")

echo > $out/rawmtx/validation.csv

BENCH_MODE=spmv
# BENCH_MODE=spmv_float

# Integer sources
declare -a bench_data=("random")
for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        echo BENCH=$BENCH_MODE LANKA=$lanka DATA_SOURCE=$source REPETITIONS=$REPETITIONS BENCH_KIND=$kind NCOLS=581012 NROWS=54 COLWISE=ON LIT_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/lit_hist.csv RUN_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/rle_hist.csv OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench

        BENCH=$BENCH_MODE LANKA=$lanka DATA_SOURCE=$source REPETITIONS=$REPETITIONS BENCH_KIND=$kind NCOLS=581012 NROWS=54 COLWISE=ON LIT_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/lit_hist.csv RUN_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/rle_hist.csv OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        
        FILE_PREFIX=$out/rawmtx/$source ; cmp --silent $(echo $FILE_PREFIX)_DENSE.tns $(echo $FILE_PREFIX)_$kind.tns && echo "Note: validation success for ${source} ${kind}!" >> $out/rawmtx/validation.csv || echo "ERROR: validation failure for ${source} ${kind}!" >> $out/rawmtx/validation.csv
        echo 
    done
    mv -f $out/rle_hist_*.csv $out/hist
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    rm $out/DENSE_spmv_COLD_*_$source.csv
done
