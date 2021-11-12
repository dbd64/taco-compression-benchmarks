#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/sketch_alpha/
lanka=OFF
mkdir -p "$out"

imgs="$ARTIFACT_DIR/data/sketches/"

mkdir -p "$out"

for i in {0..1000} 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    LANKA=OFF IMAGE_FOLDER=$imgs OUTPUT_PATH="$out" BENCH="sketch" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
done
