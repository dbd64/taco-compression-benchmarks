#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

REPETITIONS=500

while getopts s flag
do
    case "${flag}" in
        s) REPETITIONS=10
           ;;
    esac
done

if [[ ! -v SLURM_JOB_ID ]]; then
    SCRIPT_DIR=$(readlink -f $0)
elif [[ -z "$SLURM_JOB_ID" ]]; then
    SCRIPT_DIR=$(readlink -f $0)
else
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
fi

SCRIPT_DIR=/data/scratch/danielbd/artifact/tcb/scripts
ARTIFACT_DIR=$SCRIPT_DIR/../../

echo $SCRIPT_DIR

cd $SCRIPT_DIR/..

out=$ARTIFACT_DIR/out/spmv_rlep_sketches
lanka=ON
mkdir -p "$out"
mkdir -p "$out/rawmtx"
mkdir -p "$out/hist"
mkdir -p "$out/hist/graphs"

if [[ ! -v LD_LIBRARY_PATH ]]; then
    LD_LIBRARY_PATH=$SCRIPT_DIR/..
fi

# TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

# LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench
LANKA=$lanka make taco/build/taco-bench

USE_SOURCE=OFF

declare -a kinds=("DENSE" "SPARSE" "RLE" "RLEP" "LZ77")
# declare -a kinds=("DENSE" "RLEP")
declare -a kinds=("RLEP")
# declare -a kinds=("DENSE" "SPARSE" "RLE")
# declare -a kinds=("DENSE" "SPARSE" "RLE")

echo > $out/rawmtx/validation.csv

# Integer sources
declare -a bench_data=("mnist" "sketches" "hwd_plus" "covtype" "census" "spgemm" "ilsvrc" "poker")
# declare -a bench_data=("mnist" "covtype")
declare -a bench_data=("sketches" )
# declare -a bench_data=("census" "spgemm" "poker")
# declare -a bench_data=("spgemm")
# declare -a bench_data=("hwd_plus")
# declare -a bench_data=("mnist" "sketches" "hwd_plus" "ilsvrc")
# declare -a bench_data=("mnist" "covtype")
for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        # BENCH=spmv LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang CACHE_KERNELS=0 make taco-bench-nodep
        # BENCH=spmv LANKA=$lanka DATA_SOURCE=$source REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 TMPDIR=$(pwd)/build_dirs make taco-bench-nodep
        echo BENCH=spmv LANKA=$lanka DATA_SOURCE=$source USE_SOURCE=$USE_SOURCE REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        BENCH=spmv LANKA=$lanka DATA_SOURCE=$source USE_SOURCE=$USE_SOURCE REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        
        FILE_PREFIX=$out/rawmtx/$source ; cmp --silent $(echo $FILE_PREFIX)_DENSE.tns $(echo $FILE_PREFIX)_$kind.tns && echo "Note: validation success for ${source} ${kind}!" >> $out/rawmtx/validation.csv || echo "ERROR: validation failure for ${source} ${kind}!" >> $out/rawmtx/validation.csv
        FILE_PREFIX=$out/rawmtx/$source ; cmp --silent $(echo $FILE_PREFIX)_DENSE.tns $(echo $FILE_PREFIX)_$kind.tns && echo "Note: validation success for ${source} ${kind}!" || echo "ERROR: validation failure for ${source} ${kind}!"
    done
    mv -f $out/rle_hist_*.csv $out/hist
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    rm $out/DENSE_spmv_COLD_*_$source.csv

    # HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/rlep_lit_hist_$source.csv $out/hist/graphs/rlep_lit_hist_$source.png
    # HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/rlep_hist_$source.csv $out/hist/graphs/rlep_hist_$source.png

    # HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/lz77_lit_$source.csv $out/hist/graphs/lz77_lit_$source.png
    # HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/lz77_run_$source.csv $out/hist/graphs/lz77_run_$source.png
done

# Floating point sources
declare -a bench_data=("kddcup" "power")
# declare -a bench_data=("power")
declare -a bench_data=( )
for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        # BENCH=spmv_float LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang CACHE_KERNELS=0 TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        echo BENCH=spmv_float LANKA=$lanka DATA_SOURCE=$source USE_SOURCE=$USE_SOURCE REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        BENCH=spmv_float LANKA=$lanka DATA_SOURCE=$source USE_SOURCE=$USE_SOURCE REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
    
        FILE_PREFIX=$out/rawmtx/$source ; cmp --silent $(echo $FILE_PREFIX)_DENSE.tns $(echo $FILE_PREFIX)_$kind.tns && echo "Note: validation success for ${source} ${kind}!" >> $out/rawmtx/validation.csv || echo "ERROR: validation failure for ${source} ${kind}!" >> $out/rawmtx/validation.csv
    done
    mv -f $out/rle_hist_*.csv $out/hist
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    rm $out/DENSE_spmv_COLD_*_$source.csv
done


python3 $SCRIPT_DIR/merge_csv.py $out/ $out/spmv_all.csv "" "" True
# rm -r $TMP_BUILD_DIR