#!/bin/bash

videosuites=('ed:scene1,scene2,scene3,scene4' 'sita:intro_flat,flat_textures,drawn_flat,images_background' 'stock:paperclips,pink_bars,ppt,rect_bkgd')

REPETITIONS=100
REPETITIONS_MOVIE=10
ALPHA_IMGS=1000
MOVIE_FRAMES=340

while getopts s flag
do
    case "${flag}" in
        s) REPETITIONS=10
           REPETITIONS_MOVIE=3
           ALPHA_IMGS=200
           MOVIE_FRAMES=10
           ;;
    esac
done

#Build opencv_bench
mkdir -p /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build
pushd /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build
cmake /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/
make
popd

# Run benchmarks
mkdir -p /home/artifact/artifact/out/opencv/
BENCH=mri REPETITIONS=$REPETITIONS /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=alpha REPETITIONS=$REPETITIONS NUM_IMGS=$ALPHA_IMGS /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench

CLIPS_ROOT=/home/artifact/artifact/data/clips
mkdir -p /home/artifact/artifact/out/opencv/brighten/
mkdir -p /home/artifact/artifact/out/opencv/subtitle/
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
        BENCH=brighten REPETITIONS=$REPETITIONS_MOVIE NUM_IMGS=$MOVIE_FRAMES PATH1=$CLIPS_ROOT/$suiteName/$clip/ NAME=$suiteName_$clip /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
        BENCH=subtitle REPETITIONS=$REPETITIONS_MOVIE NUM_IMGS=$MOVIE_FRAMES PATH1=$CLIPS_ROOT/$suiteName/$clip/ NAME=$suiteName_$clip /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
    done
done
