#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka33
#SBATCH --exclusive

# ELEPHANTS DREAM
set -u

out=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/ed/
validation=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/ed/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/scene1/"
mkdir -p "$validation/scene3/"

ROOT_FOLDER="/data/scratch/danielbd/clips/ed_1600"

for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_lz77_$START.log"
done

# SITA
set -u

out=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/sita/
validation=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/sita/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/drawn_flat/"
mkdir -p "$validation/images_background/"

ROOT_FOLDER="/data/scratch/danielbd/clips/sita"

for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" FOLDER1="$ROOT_FOLDER/drawn_flat/" FOLDER2="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" FOLDER1="$ROOT_FOLDER/drawn_flat/" FOLDER2="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" FOLDER1="$ROOT_FOLDER/drawn_flat/" FOLDER2="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/drawn_flat/" FOLDER1="$ROOT_FOLDER/drawn_flat/" FOLDER2="$ROOT_FOLDER/flat_textures/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/drawn_flat_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/images_background/" FOLDER1="$ROOT_FOLDER/images_background/" FOLDER2="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/images_background/" FOLDER1="$ROOT_FOLDER/images_background/" FOLDER2="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/images_background/" FOLDER1="$ROOT_FOLDER/images_background/" FOLDER2="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/images_background/" FOLDER1="$ROOT_FOLDER/images_background/" FOLDER2="$ROOT_FOLDER/intro_flat/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/images_background_lz77_$START.log"
done

# BARS AND STRIPES
set -u

out=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/bs/
validation=/data/scratch/danielbd/taco-compression-benchmarks2/out/brighten_frame/bs/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/scene1/"
mkdir -p "$validation/scene3/"

ROOT_FOLDER="/data/scratch/danielbd/clips/bs_1600"

for i in {0..15} 
do
    START=$(( (i * 100) + 1 ))
    END=$(( ((i+1) * 100) ))

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene1/" FOLDER1="$ROOT_FOLDER/scene1/" FOLDER2="$ROOT_FOLDER/scene2/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene1_lz77_$START.log"

    LANKA=ON BENCH_KIND=DENSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_dense_$START.log"
    LANKA=ON BENCH_KIND=SPARSE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_sparse_$START.log"
    LANKA=ON BENCH_KIND=RLE VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_rle_$START.log"
    LANKA=ON BENCH_KIND=LZ77 VALIDATION_OUTPUT_PATH="$validation/scene3/" FOLDER1="$ROOT_FOLDER/scene3/" FOLDER2="$ROOT_FOLDER/scene4/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > "$out/scene3_lz77_$START.log"
done