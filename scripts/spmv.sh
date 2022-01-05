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

# if [ -n $SLURM_JOB_ID ];  then
#     # check the original location through scontrol and $SLURM_JOB_ID
#     SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
# else
    # otherwise: started with bash. Get the real location.
    SCRIPT_DIR=$(readlink -f $0)
# fi

SCRIPT_DIR=$(dirname $SCRIPT_DIR)
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/spmv
lanka=ON
mkdir -p "$out"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
# declare -a bench_data=("covtype" "mnist" "sketches" "ilsvrc")
declare -a bench_data=("covtype")
for source in "${bench_data[@]}"
do

    for kind in "${kinds[@]}"
    do
        BENCH=spmv DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 make taco-bench-nodep
    done
done

rm -r $TMP_BUILD_DIR