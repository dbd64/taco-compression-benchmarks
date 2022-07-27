#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u

REPETITIONS=100

while getopts sbg flag
do
    case "${flag}" in
        s) REPETITIONS=10
           ;;
        b) REPETITIONS=1000
           ;;
        g) REPETITIONS=10000
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

out=$ARTIFACT_DIR/out/spmv_perf_turbo_on
lanka=ON
mkdir -p "$out"
mkdir -p "$out/rawmtx"
mkdir -p "$out/perf"
mkdir -p "$out/raw"
mkdir -p "$out/hist"
mkdir -p "$out/hist/graphs"

# TMP_BUILD_DIR="$(mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX)"

if [[ ! -v LD_LIBRARY_PATH ]]; then
    LD_LIBRARY_PATH=$SCRIPT_DIR/..
fi

# LANKA=$lanka BUILD_DIR=$TMP_BUILD_DIR make taco/build/taco-bench
LANKA=$lanka make taco/build/taco-bench

# declare -a kinds=("DENSE" "SPARSE" "RLE" "RLEP" "LZ77")
declare -a kinds=("DENSE" "SPARSE" "RLEP")

# Integer sources
# declare -a bench_data=("covtype" "mnist" "sketches" "ilsvrc" "census" "spgemm" "poker")
# declare -a bench_data=("census" "spgemm" "poker")
declare -a bench_data=("mnist")
# declare -a bench_data=("random")

# PERF_COUNTERS=branch-misses,cache-misses,cpu-cycles
# PERF_COUNTERS=LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,cpu-cycles
PERF_COUNTERS=cycle_activity.cycles_no_execute,cycle_activity.stalls_l1d_pending,cycle_activity.stalls_l2_pending,cycle_activity.stalls_ldm_pending,resource_stalls.any,branch-misses,branches,cache-references,cache-misses,dsb2mite_switches.penalty_cycles,cpu-cycles
PERF_COUNTERS=l2_rqsts.miss,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches

PERF_COUNTERS=cycle_activity.cycles_no_execute,cycle_activity.stalls_l1d_pending,cycle_activity.stalls_l2_pending,cycle_activity.stalls_ldm_pending,resource_stalls.any,branch-misses,branches,cache-references,cache-misses,dsb2mite_switches.penalty_cycles,cpu-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches


# PERF_COUNTERS=cpu/event=0x14,umask=0x1,name=Divider_ARITH_FPU_DIV_ACTIVE,period=2000003/,cpu/event=0xc5,umask=0x0,name=Branch_Mispredicts_Branch_Resteers_BR_MISP_RETIRED_ALL_BRANCHES,period=400009/pp,cpu/event=0x79,umask=0x30,edge=1,cmask=1,name=MS_Switches_IDQ_MS_SWITCHES,period=2000003/,cpu/event=0x79,umask=0x30,name=Microcode_Sequencer_IDQ_MS_UOPS,period=2000003/,cpu/event=0xc0,umask=0x1,name=Light_Operations_INST_RETIRED_PREC_DIST,period=2000003/p,cpu/event=0x85,umask=0x2,name=ITLB_Misses_ITLB_MISSES_WALK_COMPLETED,period=100003/,cpu/event=0xc3,umask=0x1,edge=1,cmask=1,name=Machine_Clears_MACHINE_CLEARS_COUNT,period=100003/,cpu/event=0xd2,umask=0x2,name=Data_Sharing_MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_HIT,period=20011/pp,cpu/event=0xd2,umask=0x4,name=Contested_Accesses_False_Sharing_MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_HITM,period=20011/pp,cpu/event=0xd2,umask=0x1,name=Contested_Accesses_MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_MISS,period=20011/pp,cpu/event=0xd3,umask=0x10,name=False_Sharing_MEM_LOAD_UOPS_LLC_MISS_RETIRED_REMOTE_HITM,period=100007/pp,cpu/event=0xd1,umask=0x40,name=L1_Bound_MEM_LOAD_UOPS_RETIRED_HIT_LFB,period=100003/pp,cpu/event=0xd1,umask=0x1,name=L1_Bound_MEM_LOAD_UOPS_RETIRED_L1_HIT,period=2000003/pp,cpu/event=0xd1,umask=0x2,name=L2_Bound_MEM_LOAD_UOPS_RETIRED_L2_HIT,period=100003/pp,cpu/event=0xd1,umask=0x4,name=L3_Hit_Latency_L3_Bound_MEM_LOAD_UOPS_RETIRED_LLC_HIT,period=50021/pp,cpu/event=0xd1,umask=0x20,name=DRAM_Bound_MEM_LOAD_UOPS_RETIRED_LLC_MISS,period=100007/pp,cpu/event=0xd0,umask=0x82,name=Store_Bound_MEM_UOPS_RETIRED_ALL_STORES,period=2000003/pp,cpu/event=0xd0,umask=0x21,name=Lock_Latency_MEM_UOPS_RETIRED_LOCK_LOADS,period=100007/pp,cpu/event=0xd0,umask=0x41,name=Split_Loads_MEM_UOPS_RETIRED_SPLIT_LOADS,period=100003/pp,cpu/event=0xd0,umask=0x42,name=Split_Stores_MEM_UOPS_RETIRED_SPLIT_STORES,period=100003/pp,cpu/event=0xd0,umask=0x11,name=DTLB_Load_MEM_UOPS_RETIRED_STLB_MISS_LOADS,period=100003/pp,cpu/event=0xd0,umask=0x12,name=DTLB_Store_MEM_UOPS_RETIRED_STLB_MISS_STORES,period=100003/pp,cpu/event=0xb7,umask=0x1,offcore_rsp=0x10003c0002,name=False_Sharing_OFFCORE_RESPONSE_DEMAND_RFO_LLC_HIT_HITM_OTHER_CORE,period=100003/,cpu/event=0xb7,umask=0x1,offcore_rsp=0x107fc20002,name=False_Sharing_OFFCORE_RESPONSE_DEMAND_RFO_LLC_MISS_REMOTE_HITM,period=100003/,cpu/event=0xc1,umask=0x80,name=Assists_OTHER_ASSISTS_ANY_WB_ASSIST,period=100003/,cpu/event=0x5e,umask=0x1,edge=1,inv=1,cmask=1,name=Fetch_Latency_RS_EVENTS_EMPTY_END,period=200003/,cpu/event=0xc2,umask=0x2,name=Retiring_UOPS_RETIRED_RETIRE_SLOTS,period=2000003/,cycles:pp

# LIT_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/lit_hist.csv RUN_HIST=/data/scratch/danielbd/artifact/tcb/scripts/resources/rle_hist.csv 

for source in "${bench_data[@]}"
do
    for kind in "${kinds[@]}"
    do
        # BENCH=spmv LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang CACHE_KERNELS=0 make taco-bench-nodep
        BENCH=spmv LANKA=$lanka DATA_SOURCE=$source REPETITIONS=$REPETITIONS BENCH_KIND=$kind NCOLS=500000 NROWS=1000 COLWISE=ON  OUTPUT_PATH=$out CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 perf record -e $PERF_COUNTERS -c 100000 -D 5000 -o $out/perf/${source}_${kind}.data /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        # BENCH=spmv LANKA=$lanka DATA_SOURCE=$source REPETITIONS=$REPETITIONS BENCH_KIND=$kind NCNCOLS=500000 NROWS=1000 COLWISE=ON  OUTPUT_PATH=$out CACHE_KERNELS=0 TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang TMPDIR=$(pwd)/build_dirs LD_LIBRARY_PATH=/data/scratch/danielbd/artifact/tcb/taco/build:$LD_LIBRARY_PATH numactl -C 0 -m 0 perf stat -D 5000 /data/scratch/danielbd/artifact/tcb/taco/build/taco-bench
        
        FILE_PREFIX=$out/rawmtx/$source ; cmp --silent $(echo $FILE_PREFIX)_DENSE.tns $(echo $FILE_PREFIX)_$kind.tns && echo "Note: validation success for ${source} ${kind}!" >> $out/rawmtx/validation.csv || echo "ERROR: validation failure for ${source} ${kind}!" >> $out/rawmtx/validation.csv
    done
    python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
    mv $out/DENSE_spmv_COLD_*_$source.csv $out/raw/

    # echo HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/rlep_lit_hist_$source.csv $out/hist/graphs/rlep_lit_hist_$source.png
    # echo HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/rlep_hist_$source.csv $out/hist/graphs/rlep_hist_$source.png

    # echo HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/lz77_lit_$source.csv $out/hist/graphs/lz77_lit_$source.png
    # echo HOME=/data/scratch/danielbd/nfs_home/ time python3 $SCRIPT_DIR/hist_gen.py $out/hist/lz77_run_$source.csv $out/hist/graphs/lz77_run_$source.png

done

# # Floating point sources
# declare -a bench_data=("kddcup")
# # declare -a bench_data=("covtype")
# for source in "${bench_data[@]}"
# do
#     for kind in "${kinds[@]}"
#     do
#         BENCH=spmv_float LANKA=$lanka DATA_SOURCE=$source BUILD_DIR=$TMP_BUILD_DIR REPETITIONS=$REPETITIONS BENCH_KIND=$kind OUTPUT_PATH=$out/ TACO_CC=/data/scratch/danielbd/llvm-project/install/bin/clang CACHE_KERNELS=0 make taco-bench-nodep
#     done
#     python3 $SCRIPT_DIR/merge_csv.py $out/ $out/$source.csv DENSE_spmv_COLD_ _$source
#     rm $out/DENSE_spmv_COLD_*_$source.csv
# done

# python3 $SCRIPT_DIR/merge_csv.py $out/ $out/spmv_all.csv "" "" True
# rm -r $TMP_BUILD_DIR