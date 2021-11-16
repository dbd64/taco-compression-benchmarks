#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
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


SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/sketch_alpha/
lanka=OFF
mkdir -p "$out"

imgs="$ARTIFACT_DIR/data/sketches/"

mkdir -p "$out"
make taco/build/taco-bench

for i in $( seq 0 $NUM_IMGS ) 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    LANKA=OFF REPETITIONS=$REPETITIONS IMAGE_FOLDER=$imgs OUTPUT_PATH="$out" BENCH="sketch" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
done

python3 $SCRIPT_DIR/merge_csv.py $out $out/../sketch_alpha.csv sketches_alpha_blending_
rm -r $out