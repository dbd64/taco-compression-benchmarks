#!/bin/bash

#Helper variables
FOLDER0="/home/artifact/artifact/data/clips/stock/paperclips/"
FOLDER1="/home/artifact/artifact/data/clips/stock/pink_bars/"
FOLDER2="/home/artifact/artifact/data/clips/stock/ppt/"
FOLDER3="/home/artifact/artifact/data/clips/stock/rect_bkgd/"
FOLDER4="/home/artifact/artifact/data/clips/ed/scene1/"
FOLDER5="/home/artifact/artifact/data/clips/ed/scene2/"
FOLDER6="/home/artifact/artifact/data/clips/ed/scene3/"
FOLDER7="/home/artifact/artifact/data/clips/ed/scene4/"
FOLDER8="/home/artifact/artifact/data/clips/sita/intro_flat/"
FOLDER9="/home/artifact/artifact/data/clips/sita/flat_textures/"
FOLDER10="/home/artifact/artifact/data/clips/sita/drawn_flat/"
FOLDER11="/home/artifact/artifact/data/clips/sita/images_background/"

NAME0="paperclips"
NAME1="pink_bars"
NAME2="ppt"
NAME3="rect_bkgd"
NAME4="ed_scene1"
NAME5="ed_scene2"
NAME6="ed_scene3"
NAME7="ed_scene4"
NAME8="sita_intro_flat"
NAME9="sita_flat_textures"
NAME10="sita_drawn_flat" 
NAME11="sita_images_background"

#Build opencv_bench
mkdir -p /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build
pushd /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build
cmake /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/
make
popd

# Run benchmarks
mkdir -p /home/artifact/artifact/out/opencv/
BENCH=mri /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=alpha /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench

mkdir -p /home/artifact/artifact/out/opencv/brighten/
BENCH=brighten PATH1=$FOLDER0 NAME=$NAME0 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER1 NAME=$NAME1 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER2 NAME=$NAME2 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER3 NAME=$NAME3 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER4 NAME=$NAME4 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER5 NAME=$NAME5 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER6 NAME=$NAME6 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER7 NAME=$NAME7 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER8 NAME=$NAME8 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER9 NAME=$NAME9 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER10 NAME=$NAME10 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=brighten PATH1=$FOLDER11 NAME=$NAME11 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench

mkdir -p /home/artifact/artifact/out/opencv/subtitle/
BENCH=subtitle PATH1=$FOLDER0 NAME=$NAME0 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER1 NAME=$NAME1 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER2 NAME=$NAME2 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER3 NAME=$NAME3 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER4 NAME=$NAME4 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER5 NAME=$NAME5 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER6 NAME=$NAME6 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER7 NAME=$NAME7 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER8 NAME=$NAME8 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER9 NAME=$NAME9 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER10 NAME=$NAME10 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
BENCH=subtitle PATH1=$FOLDER11 NAME=$NAME11 /home/artifact/artifact/taco-compression-benchmarks/opencv_bench/build/opencv-bench
