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

if [[ ! -v SLURM_JOB_ID ]]; then
    SCRIPT_DIR=$(readlink -f $0)
elif [[ -z "$SLURM_JOB_ID" ]]; then
    SCRIPT_DIR=$(readlink -f $0)
else
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
fi

SCRIPT_DIR=$(dirname $SCRIPT_DIR)
ARTIFACT_DIR=$SCRIPT_DIR/../../

echo $SCRIPT_DIR

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/spmv
lanka=ON
mkdir -p "$out"
mkdir -p "$out/rawmtx"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench

# declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
declare -a kinds=("DENSE" "SPARSE" "RLE")

# Integer sources
declare -a bench_data=("covtype" "mnist" "sketches" "ilsvrc" "census" "spgemm" "poker")
# declare -a bench_data=("census" "spgemm" "poker")
# declare -a bench_data=("poker")
for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        BENCH=spmv LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 make taco-bench-nodep
    done
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    rm $out/DENSE_spmv_COLD_*_$source.csv
done

# Floating point sources
declare -a bench_data=("kddcup")
# declare -a bench_data=("covtype")
for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        BENCH=spmv_float LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 make taco-bench-nodep
    done
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    rm $out/DENSE_spmv_COLD_*_$source.csv
done

rm -r $TMP_BUILD_DIR