#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/bs/
validation=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/bs/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/title/"
mkdir -p "$validation/scene1/"
mkdir -p "$validation/scene2"
mkdir -p "$validation/scene3"


for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH="brighten" BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/title/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/title_dense_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/title/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/title_sparse_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/title/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/title_rle_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/title/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/title_lz77_$START.log"

    LANKA=ON BENCH="brighten" BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene1/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench > "$out/scene1_dense_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene1/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench > "$out/scene1_sparse_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene1/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench > "$out/scene1_rle_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene1/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench > "$out/scene1_lz77_$START.log"

    LANKA=ON BENCH="brighten" BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene2_dense_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene2_sparse_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene2_rle_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene2_lz77_$START.log"

    LANKA=ON BENCH="brighten" BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene3_dense_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene3_sparse_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene3_rle_$START.log"
    LANKA=ON BENCH="brighten" BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs_1600/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench > "$out/scene3_lz77_$START.log"
done