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

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_DIR=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_DIR)
ARTIFACT_DIR=$SCRIPT_DIR/../../

#Build opencv_bench
mkdir -p $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build
pushd $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build
cmake $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/
make
popd

# Run benchmarks
mkdir -p $ARTIFACT_DIR/out/opencv/
BENCH=mri REPETITIONS=$REPETITIONS $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=alpha REPETITIONS=$REPETITIONS NUM_IMGS=$ALPHA_IMGS $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build/opencv-bench

CLIPS_ROOT=$ARTIFACT_DIR/data/clips
mkdir -p $ARTIFACT_DIR/out/opencv/brighten/
mkdir -p $ARTIFACT_DIR/out/opencv/subtitle/
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
        BENCH=brighten REPETITIONS=$REPETITIONS_MOVIE NUM_IMGS=$MOVIE_FRAMES PATH1=$CLIPS_ROOT/$suiteName/$clip/ NAME=$suiteName_$clip $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build/opencv-bench
        BENCH=subtitle REPETITIONS=$REPETITIONS_MOVIE NUM_IMGS=$MOVIE_FRAMES PATH1=$CLIPS_ROOT/$suiteName/$clip/ NAME=$suiteName_$clip $ARTIFACT_DIR/taco-compression-benchmarks/opencv_bench/build/opencv-bench
    done
done
