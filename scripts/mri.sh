#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

out=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/out/roi/ #/data/scratch/danielbd/taco-compression-benchmarks/out/roi/
# validation=/data/scratch/danielbd/taco-compression-benchmarks/out/roi/validation/
lanka=OFF
mkdir -p "$out"
# mkdir -p "$validation"

LANKA=$lanka IMAGE_FOLDER="" BENCH_KIND="DENSE" CACHE_KERNELS=0 make taco-bench > $out/dense.out
LANKA=$lanka IMAGE_FOLDER="" BENCH_KIND="SPARSE" CACHE_KERNELS=0 make taco-bench > $out/sparse.out
LANKA=$lanka IMAGE_FOLDER="" BENCH_KIND="RLE" CACHE_KERNELS=0 make taco-bench > $out/rle.out
LANKA=$lanka IMAGE_FOLDER="" BENCH_KIND="LZ77" CACHE_KERNELS=0 make taco-bench > $out/lz77.out