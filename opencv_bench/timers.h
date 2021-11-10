#ifndef TACO_UTIL_BENCHMARK_H
#define TACO_UTIL_BENCHMARK_H


#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>



using namespace std;

template <class Tp>
inline __attribute__((always_inline)) void DoNotOptimize(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

template <class Tp>
inline __attribute__((always_inline)) void DoNotOptimizePtr(Tp* value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

struct TimeResults {
  double mean;
  double stdev;
  double median;
  vector<double> times;
  int size;

  friend std::ostream& operator<<(std::ostream& os, const TimeResults& t) {
    if (t.size == 1) {
      return os << t.mean;
    }
    else {
      os << t.mean << "," << t.stdev << "," << t.median << ",";
      for (int i = 0; i<t.times.size()-1; i++){
        os << t.times[i] << ",";
      }
      return os << t.times[t.times.size()-1];
//      return os << "  mean:   " << t.mean   << endl
//                << "  stdev:  " << t.stdev  << endl
//                << "  median: " << t.median;
    }
  }
};

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;

/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:
  void start() {
    begin = std::chrono::high_resolution_clock::now(); // high resolution clock
  }

  void stop() {
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    times.push_back(diff);
  }

  // Compute mean, standard deviation and median
  TimeResults getResult() {
    // JUST OUTPUT RAW DATA

    int repeat = static_cast<int>(times.size());

    TimeResults result;
    result.times = times;
    double mean=0.0;
    // times = ends - begins
    sort(times.begin(), times.end());
    // remove 10% worst and best cases
    const int truncate = 0;
    mean = accumulate(times.begin() + truncate, times.end() - truncate, 0.0);
    int size = repeat - 2 * truncate;
    result.size = size;
    mean = mean/size;
    result.mean = mean;

    vector<double> diff(size);
    transform(times.begin() + truncate, times.end() - truncate,
              diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = inner_product(diff.begin(), diff.end(),
                                  diff.begin(), 0.0);
    result.stdev = std::sqrt(sq_sum / size);
    result.median = (size % 2)
                    ? times[size/2]
                    : (times[size/2-1] + times[size/2]) / 2;
    return result;
  }

  double __attribute__ ((noinline)) clear_cache() {
    double ret = 0.0;
    if (!dummyA) {
      dummyA = (double*)(malloc(dummySize*sizeof(double)));
      dummyB = (double*)(malloc(dummySize*sizeof(double)));
    }
    for (int i=0; i< 100; i++) {
      dummyA[rand() % dummySize] = rand()/RAND_MAX;
      dummyB[rand() % dummySize] = rand()/RAND_MAX;
    }
    for (int i=0; i<dummySize; i++) {
      ret += dummyA[i] * dummyB[i];
    }
    return ret;
  }

  double* dummy(){
    return (rand() > RAND_MAX/2) ? dummyA : dummyB;
  }

protected:
  vector<double> times;
  TimePoint begin;
private:
  int dummySize = 30000000;
  double* dummyA = NULL;
  double* dummyB = NULL;
};

#endif