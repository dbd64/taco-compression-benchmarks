#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka27
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/sita/
validation=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten_frame/sita/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/intro_flat/"
mkdir -p "$validation/flat_textures/"
mkdir -p "$validation/drawn_flat/"
mkdir -p "$validation/images_background/"

ROOT_FOLDER="/data/scratch/danielbd/clips/sita"

for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/intro_flat/" IMAGE_FOLDER="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/intro_flat_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/intro_flat/" IMAGE_FOLDER="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/intro_flat_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/intro_flat/" IMAGE_FOLDER="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/intro_flat_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/intro_flat/" IMAGE_FOLDER="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/intro_flat_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/flat_textures/" IMAGE_FOLDER="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/flat_textures_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/flat_textures/" IMAGE_FOLDER="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/flat_textures_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/flat_textures/" IMAGE_FOLDER="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/flat_textures_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/flat_textures/" IMAGE_FOLDER="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END  CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/flat_textures_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" IMAGE_FOLDER="$ROOT_FOLDER/drawn_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" IMAGE_FOLDER="$ROOT_FOLDER/drawn_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" IMAGE_FOLDER="$ROOT_FOLDER/drawn_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" IMAGE_FOLDER="$ROOT_FOLDER/drawn_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/images_background/" IMAGE_FOLDER="$ROOT_FOLDER/images_background/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/images_background/" IMAGE_FOLDER="$ROOT_FOLDER/images_background/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/images_background/" IMAGE_FOLDER="$ROOT_FOLDER/images_background/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/images_background/" IMAGE_FOLDER="$ROOT_FOLDER/images_background/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_lz77_$START.log"
done