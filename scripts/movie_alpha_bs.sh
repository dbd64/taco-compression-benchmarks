#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka32
#SBATCH --exclusive

# BARS AND STRIPES
set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/alpha/bs/
validation=$out/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/scene1/"
mkdir -p "$validation/scene3/"

ROOT_FOLDER="/data/scratch/danielbd/clips/bs_1600"

for i in {0..159} 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/title/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_lz77_$START.log"
done