#!/bin/bash

REPETITIONS=10
MOVIE_FRAMES=33

while getopts s flag
do
    case "${flag}" in
        s) REPETITIONS=3
           MOVIE_FRAMES=0
           ;;
    esac
done

set -u

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_DIR=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_DIR)
ARTIFACT_DIR=$SCRIPT_DIR/../../
CLIPS_DIR=$ARTIFACT_DIR/data/clips/

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/movie/
mkdir -p "$out"

videosuites=('ed:scene1,scene2,scene3,scene4' 'sita:intro_flat,flat_textures,drawn_flat,images_background' 'stock:paperclips,pink_bars,ppt,rect_bkgd')
declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
declare -a kinds_compressed=("RLE" "LZ77")

make taco/build/taco-bench

for suite in ${videosuites[@]}
do
    unset clipsList
    if [[ $suite == *":"* ]]
    then
        tmpArray=(${suite//:/ })
        suiteName=${tmpArray[0]}
        clipsList=${tmpArray[1]}
        clipsList=(${clipsList//,/ })
    fi

    for clip in ${clipsList[@]}
    do
        mkdir -p $out/subtitle/$suiteName/
        mkdir -p $out/brighten/$suiteName/

        mkdir -p $out/decompress/$suiteName/
        mkdir -p $out/brighten_compress/$suiteName/
        mkdir -p $out/subtitle_compress/$suiteName/
        mkdir -p $out/lz77_to_rle/$suiteName/
        mkdir -p $out/lz77_to_rle_brighten/$suiteName/
        mkdir -p $out/lz77_to_rle_subtitle/$suiteName/

        for i in $( seq 0 $MOVIE_FRAMES )
        do
            START=$(( (i * 10) + 1 ))
            END=$(( ((i+1) * 10) ))

            for kind in "${kinds[@]}"
            do
                #Computation benchmarks
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=subtitle BENCH_KIND=$kind OUTPUT_PATH="$out/subtitle/$suiteName/" FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=mbrighten BENCH_KIND=$kind OUTPUT_PATH="$out/brighten/$suiteName/" FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
            done

            for kind in "${kinds_compressed[@]}"
            do
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=decompress BENCH_KIND=$kind OUTPUT_PATH="$out/decompress/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=brighten_compress BENCH_KIND=$kind OUTPUT_PATH="$out/brighten_compress/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=subtitle_compress BENCH_KIND=$kind OUTPUT_PATH="$out/subtitle_compress/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
            
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=lz77_rle BENCH_KIND=$kind OUTPUT_PATH="$out/lz77_to_rle/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=lz77_rle_brighten BENCH_KIND=$kind OUTPUT_PATH="$out/lz77_to_rle_brighten/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
                LANKA=OFF REPETITIONS=$REPETITIONS BENCH=lz77_rle_subtitle BENCH_KIND=$kind OUTPUT_PATH="$out/lz77_to_rle_subtitle/$suiteName/"  FOLDER="$CLIPS_DIR$suiteName/$clip/" IMAGE_START=$START IMAGE_END=$END CACHE_KERNELS=0 make taco-bench-nodep 
            done

        done
    done
done
