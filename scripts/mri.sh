#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka29
#SBATCH --exclusive

set -u

SCRIPT_DIR=$(dirname $(readlink -f $0))
ARTIFACT_DIR=$SCRIPT_DIR/../../

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/mri_temp/
lanka=OFF
mkdir -p "$out"

imgs="$ARTIFACT_DIR/taco-compression-benchmarks/data/mri/"

for i in {1..253} 
do
    START=$(( i ))
    END=$(( i ))
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench 
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" BENCH="mri" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench
done

python3 $SCRIPT_DIR/merge_csv.py $out $out/../mri.csv

rm -r $out