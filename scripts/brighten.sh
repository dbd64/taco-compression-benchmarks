#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten/
validation=/data/scratch/danielbd/taco-compression-benchmarks/out/brighten/validation/
mkdir -p "$out"
mkdir -p "$validation"
mkdir -p "$validation/character1/"
mkdir -p "$validation/character2/"
mkdir -p "$validation/merrie_melodies"

LANKA=ON VALIDATION_OUTPUT_PATH="$validation/character1/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs/character1/" IMAGE_START=1602 IMAGE_END=2201 CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > $out/character1.log

LANKA=ON VALIDATION_OUTPUT_PATH="$validation/character2/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs/character2/" IMAGE_START=3023 IMAGE_END=3622  CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > $out/character2.log

LANKA=ON VALIDATION_OUTPUT_PATH="$validation/merrie_melodies/" IMAGE_FOLDER="/data/scratch/danielbd/clips/bs/merrie_melodies/" IMAGE_START=1 IMAGE_END=600 CACHE_KERNELS=0 make taco-bench BENCHES="brighten" > $out/merrie_melodies.log
