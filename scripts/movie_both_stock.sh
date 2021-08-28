#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka33
#SBATCH --exclusive

# BARS AND STRIPES
set -u

out=/data/scratch/danielbd/out_lz77_rle/stock/
validation=$out/validation/
mkdir -p "$out"
compiler="/data/scratch/danielbd/llvm-project/install/bin/clang -march=native"
ROOT_FOLDER="/data/scratch/danielbd/clips/stock_videos/"

# declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
declare -a kinds=("RLE" "LZ77")
# declare -a folders=("paperclips" "pink_bars" "ppt" "rect_bkgd" "earth" "whale" "pencils")
# declare -a folders=("paperclips" "pink_bars" "ppt" "rect_bkgd")
# declare -a folders=( "earth" "whale" "pencils")

declare -a kinds=("LZ77")
declare -a folders=("paperclips" "rect_bkgd")

# for i in {40..44}
# for i in {45..49}
# for i in {50..54}
# for i in {54..59}
# for i in {60..64} # 31
for i in {65..67} # 32
do
    START=$(( (i * 5) + 1 ))
    END=$(( ((i+1) * 5) ))

    for kind in "${kinds[@]}"
    do
        for clip in "${folders[@]}"
        do 
            # TACO_CC="$compiler" LANKA=ON BENCH=subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=mbrighten BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=decompress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=brighten_compress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            # TACO_CC="$compiler" LANKA=ON BENCH=subtitle_compress BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 

            TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle_brighten BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
            TACO_CC="$compiler" LANKA=ON BENCH=lz77_rle_subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation/$clip/" FOLDER="$ROOT_FOLDER/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
        done
    done
done
