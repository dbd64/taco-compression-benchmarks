#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka29
#SBATCH --exclusive

# BARS AND STRIPES
set -u

out=/data/scratch/danielbd/out_small_test/
validation=/data/scratch/danielbd/out_small_test/validation/
mkdir -p "$out"
ROOT_FOLDER="/data/scratch/danielbd/clips/stock_videos"

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")

# for i in {0..35} 
for i in {0..0} 
do
    START=$(( (i * 10) + 1 ))
    END=$(( ((i+1) * 10) ))

    for kind in "${kinds[@]}"
    do
        # TACO_CC="/data/scratch/danielbd/llvm-project/install/bin/clang -march=native" LANKA=ON BENCH=subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" FOLDER1="$ROOT_FOLDER/paperclips/" FOLDER2="$ROOT_FOLDER/rect_bkgd/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 
        
        # LANKA=ON BENCH=alpha BENCH_KIND=$kind OUTPUT_PATH="$out"
        # VALIDATION_OUTPUT_PATH="$validation" FOLDER1="$ROOT_FOLDER/scene3/"
        # FOLDER2="$ROOT_FOLDER/title/" IMAGE_START=$START IMAGE_END=$END
        # CACHE_KERNELS=0 make taco-bench 
        
        TACO_CC="/data/scratch/danielbd/llvm-project/install/bin/clang -march=native" LANKA=ON BENCH=subtitle BENCH_KIND=$kind OUTPUT_PATH="$out" VALIDATION_OUTPUT_PATH="$validation" FOLDER="$ROOT_FOLDER/paperclips/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench 

        TACO_CC="/data/scratch/danielbd/llvm-project/install/bin/clang -march=native" LANKA=ON BENCH=subtitle BENCH_KIND=DENSE OUTPUT_PATH="/data/scratch/danielbd/out_small_test/" VALIDATION_OUTPUT_PATH="/data/scratch/danielbd/out_small_test/validation/" FOLDER="$/data/scratch/danielbd/clips/stock_videos/paperclips/" IMAGE_START=1 IMAGE_END=1 CACHE_KERNELS=0

    done
done