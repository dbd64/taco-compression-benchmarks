#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka31
#SBATCH --exclusive

# BARS AND STRIPES
set -u

out=/data/scratch/danielbd/out_lz77_rle/ed/
validation=$out/validation/
mkdir -p "$out"
compiler="/data/scratch/danielbd/llvm-project/install/bin/clang -march=native"
ROOT_FOLDER="/data/scratch/danielbd/clips/ed_1600/"

# declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
declare -a kinds=("RLE" "LZ77")
declare -a folders=("scene1" "scene2" "scene3" "scene4")

for i in {0..33} 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    for kind in "${kinds[@]}"
    do
        for clip in "${folders[@]}"
        do 
            # TACO_CC="$compiler" LANKA=ON BENCH=subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=mbrighten BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=decompress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=brighten_compress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=subtitle_compress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
        
            # TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle_brighten BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle_subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
        done
    done
done