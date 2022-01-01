#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

RU_UPPER_BOUND=100
NUM_RAND=10
RUN_MULTIPLIER=10
REPETITIONS=100
RAND_WIDTH=1000
RAND_HEIGHT=10000

while getopts s flag
do
    case "${flag}" in
        s) RU_UPPER_BOUND=10
           NUM_RAND=3
           RUN_MULTIPLIER=40
           REPETITIONS=10
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

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/micro
lanka=ON
mkdir -p "$out/constmul/"
mkdir -p "$out/elemwise/"
mkdir -p "$out/maskmul/"
mkdir -p "$out/spmv/"

TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench

declare -a kinds=("DENSE" "SPARSE" "RLE" "LZ77")
for ru in $( seq 1 $RU_UPPER_BOUND ) 
do
    RUNUP=$(( (ru * $RUN_MULTIPLIER) ))

    for kind in "${kinds[@]}"
    do
        for i in $( seq 0 $(( ($NUM_RAND - 1) )) )
        do
            BENCH=micro_constmul BUILD_DIR=$TMP_BUILD_DIR RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/constmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
            BENCH=micro_elemwise BUILD_DIR=$TMP_BUILD_DIR RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/elemwise/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep          
            BENCH=micro_maskmul BUILD_DIR=$TMP_BUILD_DIR RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/maskmul/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
            BENCH=spmv_rand BUILD_DIR=$TMP_BUILD_DIR RAND_WIDTH=$RAND_WIDTH RAND_HEIGHT=$RAND_HEIGHT REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/spmv/ RUN_UPPER=$RUNUP INDEX=$i CACHE_KERNELS=0 make taco-bench-nodep
        done
    done
done

python3 $SCRIPT_DIR/merge_csv.py $out/constmul $out/constmul.csv
python3 $SCRIPT_DIR/merge_csv.py $out/elemwise $out/elemwise.csv
python3 $SCRIPT_DIR/merge_csv.py $out/maskmul $out/maskmul.csv
python3 $SCRIPT_DIR/merge_csv.py $out/spmv $out/spmv.csv
rm -r "$out/constmul/"
rm -r "$out/elemwise/"
rm -r "$out/maskmul/"
rm -r "$out/spmv/"
