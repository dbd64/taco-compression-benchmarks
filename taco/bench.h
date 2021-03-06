#ifndef TACO_BENCH_BENCH_H
#define TACO_BENCH_BENCH_H

#include "benchmark/benchmark.h"
#include "taco/tensor.h"

// Register a benchmark with the following options:
// * Millisecond output display
// * 10 data points
// * Reporting of avg/stddev/median
// * Wall-clock time, rather than CPU time.
#define TACO_BENCH(bench)         \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(10)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(true)    \
  ->UseRealTime()

#define GRAPHBLAS_BENCH(bench,times)   \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(times)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(false)    \
  ->UseRealTime()

// TACO_BENCH_ARG is similar to TACO_BENCH but allows for passing
// of an arbitrarily typed argument to the benchmark function.
// TODO (rohany): Make this take in only 1 argument.
// TODO (rohany): Don't specify the time here, but do it at the command line.

#define TACO_BENCH_ARG(bench, name, arg)  \
  BENCHMARK_CAPTURE(bench, name, arg)     \
  ->Unit(benchmark::kMicrosecond)         \
  ->Iterations(1)                         \
  ->ReportAggregatesOnly(true)            \
  ->UseRealTime()

#define TACO_BENCH_ARGS(bench, name, ...)       \
  BENCHMARK_CAPTURE(bench, name, __VA_ARGS__)   \
  ->Unit(benchmark::kMicrosecond)               \
  ->Iterations(1)                               \
  ->ReportAggregatesOnly(true)                  \
  ->UseRealTime()

std::string getEnvVar(std::string varname);
std::string getTacoTensorPath();
std::string getOutputPath();
std::string getValidationOutputPath();
// cleanPath ensures that the input path ends with "/".
std::string cleanPath(std::string path);
#endif //TACO_BENCH_BENCH_H
