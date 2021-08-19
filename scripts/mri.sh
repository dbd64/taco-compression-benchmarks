#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka29
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/out_all/
validation=/data/scratch/danielbd/taco-compression-benchmarks2/out/roi_sparse_change/validation/
lanka=ON
mkdir -p "$out"
mkdir -p "$validation"

imgs="/data/scratch/danielbd/taco-compression-benchmarks2/data/mri/"

for i in {0..24} 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench 
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench
    LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench
done

START=251
END=253

LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench
LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench 
LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench
LANKA=$lanka IMAGE_START=$START IMAGE_END=$END IMAGE_FOLDER="$imgs" OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench
