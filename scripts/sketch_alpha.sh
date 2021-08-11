#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --nodelist=lanka26
#SBATCH --exclusive

set -u

out=/data/scratch/danielbd/taco-compression-benchmarks/out/sketch_alpha/
mkdir -p "$out"
LANKA=ON IMAGE_FOLDER="/data/scratch/danielbd/python_png_analysis/sketches/nodelta/" make taco-bench BENCHES="sketch_alpha"