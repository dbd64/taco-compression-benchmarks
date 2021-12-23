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

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_DIR=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_DIR)
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/mri_temp/
lanka=ON
mkdir -p "$out"

imgs="$SCRIPT_DIR/../data/mri/"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench
for i in {1..253} 
do
    START=$(( i ))
    END=$(( i ))
    LANKA=$lanka REPETITIONS=$REPETITIONS BUILD_DIR=$TMP_BUILD_DIR IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench-nodep
    LANKA=$lanka REPETITIONS=$REPETITIONS BUILD_DIR=$TMP_BUILD_DIR IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench-nodep 
    LANKA=$lanka REPETITIONS=$REPETITIONS BUILD_DIR=$TMP_BUILD_DIR IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench-nodep
    LANKA=$lanka REPETITIONS=$REPETITIONS BUILD_DIR=$TMP_BUILD_DIR IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench-nodep
done

python3 $SCRIPT_DIR/merge_csv.py $out $out/../mri.csv

rm -r $out
rm -r $TMP_BUILD_DIR