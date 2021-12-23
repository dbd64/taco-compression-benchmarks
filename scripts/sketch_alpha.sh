#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

NUM_IMGS=100
REPETITIONS=100

while getopts s flag
do
    case "${flag}" in
        s) NUM_IMGS=20
           REPETITIONS=10
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

out=$ARTIFACT_DIR/out/sketch_alpha/
lanka=ON
mkdir -p "$out"

imgs="$ARTIFACT_DIR/data/sketches/"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

mkdir -p "$out"

LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench
for i in $( seq 0 $NUM_IMGS ) 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    LANKA=OFF REPETITIONS=$REPETITIONS BUILD_DIR=$TMP_BUILD_DIR IMAGE_FOLDER=$imgs OUTPUT_PATH="$out" BENCH="sketch" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
done

python3 $SCRIPT_DIR/merge_csv.py $out $out/../sketch_alpha.csv sketches_alpha_blending_
rm -r $out
rm -r $TMP_BUILD_DIR