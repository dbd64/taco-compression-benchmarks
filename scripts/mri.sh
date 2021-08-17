#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka29
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks2/out/roi_sparse_change/
validation=/data/scratch/danielbd/taco-compression-benchmarks2/out/roi_sparse_change/validation/
lanka=ON
mkdir -p "$out"
mkdir -p "$validation"

imgs="/data/scratch/danielbd/taco-compression-benchmarks2/data/mri/"

LANKA=$lanka IMAGE_FOLDER="$imgs" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench > $out/sparse.out
LANKA=$lanka IMAGE_FOLDER="$imgs" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench > $out/dense.out
LANKA=$lanka IMAGE_FOLDER="$imgs" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench > $out/rle.out
LANKA=$lanka IMAGE_FOLDER="$imgs" VALIDATION_OUTPUT_PATH="$validation" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench > $out/lz77.out