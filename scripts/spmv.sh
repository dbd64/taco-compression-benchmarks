# BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench


# CACHE=WARM BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# CACHE=WARM BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# CACHE=WARM BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=covtype OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# CACHE=WARM BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# CACHE=WARM BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
# CACHE=WARM BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=mnist OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# CACHE=WARM BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
CACHE=WARM BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
CACHE=WARM BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=sketches OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench

# CACHE=WARM BENCH=spmv BENCH_KIND=DENSE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
CACHE=WARM BENCH=spmv BENCH_KIND=SPARSE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
CACHE=WARM BENCH=spmv BENCH_KIND=RLE DATA_SOURCE=ilsvrc OUTPUT_PATH=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/OUT_TEST/ CACHE_KERNELS=0 make taco-bench
