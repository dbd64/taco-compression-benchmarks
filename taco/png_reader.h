#ifndef PNG_READER_H
#define PNG_READER_H

#include "taco.h"
#include "taco/tensor.h"
#include "lodepng.h"
#include <iterator>

using namespace taco;

#define TACO_TIME_REPEAT(CODE, REPEAT, RES, COLD) {  \
    taco::util::Timer timer;                         \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
    }                                                \
    RES = timer.getResult();                         \
  }

#define TOOL_BENCHMARK_REPEAT(CODE, NAME, REPEAT) {              \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,REPEAT,timevalue,false);             \
      cout << timevalue << endl; \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_TIMER(CODE,NAME,TIMER) {                  \
    if (time) {                                                  \
      taco::util::Timer timer;                                   \
      timer.start();                                             \
      CODE;                                                      \
      timer.stop();                                              \
      taco::util::TimeResults result = timer.getResult();        \
      cout << NAME << " " << result << " ms" << endl;            \
      TIMER=result;                                              \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

enum class Kind {
      DENSE,
      SPARSE,
      RLE,
      LZ77
};

Func getCopyFunc();
Func getPlusFunc();
Func getPlusRleFunc();
Kernel getKernel(IndexStmt indexStmt, Tensor<uint8_t> t);

std::pair<Tensor<uint8_t>, size_t> read_png(int i, Kind kind);
std::pair<Tensor<uint8_t>, size_t> read_rgb_png(int i, Kind kind);

std::vector<uint8_t> unpackLZ77_bytes(std::vector<uint8_t> bytes);

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path);


unsigned decode(std::vector<unsigned char>& out, std::vector<unsigned char>& c_out, std::vector<int>& pos, unsigned& w, unsigned& h,
                const std::vector<unsigned char>& in, LodePNGColorType colortype = LCT_GREY, unsigned bitdepth = 8);

#endif