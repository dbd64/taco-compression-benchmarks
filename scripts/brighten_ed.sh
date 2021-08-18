#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka28
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/ed/
validation=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/ed/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/scene1/"
mkdir -p "$validation/scene2/"
mkdir -p "$validation/scene3/"
mkdir -p "$validation/scene4/"

ROOT_FOLDER="/data/scratch/danielbd/clips/ed_1600"

for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="$ROOT_FOLDER/scene1/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="$ROOT_FOLDER/scene1/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="$ROOT_FOLDER/scene1/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene1/" IMAGE_FOLDER="$ROOT_FOLDER/scene1/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene2_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene2_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene2_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene2/" IMAGE_FOLDER="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene2_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="$ROOT_FOLDER/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="$ROOT_FOLDER/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="$ROOT_FOLDER/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene3/" IMAGE_FOLDER="$ROOT_FOLDER/scene3/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene4/" IMAGE_FOLDER="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene4_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene4/" IMAGE_FOLDER="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene4_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene4/" IMAGE_FOLDER="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene4_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene4/" IMAGE_FOLDER="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene4_lz77_$START.log"
done