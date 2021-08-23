#include "bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"
#include "taco/util/timers.h"
#include "utils.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"
#include "png_reader.h"

using namespace taco;

#include "png_reader.h"

// 0 XXXXXXX XXXXXXXX     -> read X number of bytes
// 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

#define MAX_LIT  32767
#define MAX_RUN  32767
#define MAX_DIST 65535

namespace {
__attribute__((always_inline))
int computeHash(const uint8_t* in, const int in_idx){
  return ( ( in[ in_idx ] & 0xFF ) << 8 ) |
         ( ( in[ in_idx + 1 ] & 0xFF ) ^ ( in[ in_idx + 2 ] & 0xFF ) );
}

__attribute__((always_inline))
std::pair<int,int> find_match(const int in_idx, const int out_idx, const int start_idx,
                              const uint8_t* in, int insize, const uint8_t* out, int outsize) {
  int len = 1;
  int off = out_idx - start_idx;
  if( off > 0 && off < 65536 && start_idx >= 0 && start_idx < outsize &&
      out[ start_idx ] == in[ in_idx ] ) {
    // while( in_idx + len < insize && len <= MAX_RUN && 
    //        out[ start_idx + (len % off) ] == in[ in_idx + len ] ) {
    //   len++;
    // }
    int out_len = len;
    while( in_idx + len < insize && len <= MAX_RUN && 
           out[ start_idx + out_len ] == in[ in_idx + len ] ) {
      len++;
      out_len++;
      if (out_len == off) out_len = 0;
    }
  }
  return {len, off};
}

__attribute__((always_inline))
void push_uint16(uint8_t* out, int& outsize, int& outcap, uint16_t value){
  if (outsize >= outcap){
    outcap *= 2;
    out = (uint8_t*)realloc((void*)out, outcap);
  }
  uint8_t bytes[sizeof(value)];
  memcpy(bytes, &value, sizeof(value));
  out[outsize+0] = bytes[0];
  out[outsize+1] = bytes[1];
  outsize+=2;
}

__attribute__((always_inline))
void set_uint16(uint8_t* out, int& outsize, int& outcap, const int index, uint16_t value){
  uint8_t bytes[sizeof(value)];
  memcpy(bytes, &value, sizeof(value));

  out[index + 0] = bytes[0];
  out[index + 1] = bytes[1];
}

__attribute__((always_inline))
uint16_t load_uint16(const uint8_t* out, int& outsize, int& outcap, const int index){
  uint16_t val;
  memcpy(&val, &out[index], sizeof(val));
  return val;
}

__attribute__((noinline))
void encode_lz77_(const uint8_t* in, const int insize, uint8_t*& out, int& outsize) {
  int in_idx = 0;
  int count_idx = -1;
  int count = 0;
  // std::vector<int> hash(65536);
  int outcap = 1048576;
  out = (uint8_t*) malloc(outcap);
  outsize = 0;

  while( in_idx < insize ) {
    int len = 1;
    int off = 0;
    if( in_idx + 2 < insize ) {
      auto pixelRle = find_match(in_idx, outsize, outsize - 3, in, insize, out, outsize);
      len = pixelRle.first;
      off = pixelRle.second;
    }
    if( len >= 4 ) {
      push_uint16(out, outsize, outcap, 32768 | len);
      push_uint16(out, outsize, outcap, off);
      if (count_idx != -1){
        set_uint16(out, outsize, outcap, count_idx, count);
        count_idx = -1;
      }
      // hash[computeHash(out, outsize-4)] = outsize-4;
      // hash[computeHash(out, outsize-3)] = outsize-3;
      in_idx+=len;
    } else {
      while( len ) {
        if (count_idx == -1) {
          count_idx = outsize;
          count = 0;
          push_uint16(out, outsize, outcap, 0);
        }
        
        int old_count = count; //load_uint16(out, outsize, outcap, count_idx);
        if (old_count == MAX_LIT){
          set_uint16(out, outsize, outcap, count_idx, old_count);
          count_idx = outsize;
          push_uint16(out, outsize, outcap, 0);
        }

        int new_count = len+old_count >= MAX_LIT ? MAX_LIT : len+old_count;
        int max = new_count - old_count;

        len -= max;
        //set_uint16(out, outsize, outcap, count_idx, new_count);
        count = new_count;
        // if (count_idx+2<outsize) {
        //   hash[computeHash(out, count_idx)] = count_idx;
        // }
        int outindex = outsize;
        outsize+=max;
        if (outsize >= outcap){
          outcap = outcap*2;
          out = (uint8_t*)realloc((void*)out, outcap);
        }
        while( max ) {
          out[outindex++] = in[ in_idx++ ];
          // if(outsize > 2) {
          //   hash[computeHash(out, outsize-3)] = outsize-3;
          // }
          max--;
        }
      }
    }
  }
}
}

namespace {
  Func getBrightenFunc(uint8_t brightness, bool full){
    auto brighten = [=](const std::vector<ir::Expr>& v) {
        auto sum = ir::Add::make(v[0], brightness);
        return ternaryOp(ir::Gt::make(sum, 255), 255, sum);
    };
    auto algFunc = [=](const std::vector<IndexExpr>& v) {
        auto l = Region(v[0]);
        if (full){
          return IterationAlgebra(Union(l, Complement(l)));
        } else {
          return IterationAlgebra(l);
        }
    };
    Func plus_("plus_", brighten, algFunc);
    return plus_;
  }
}

namespace{
void writeHeaderRecompress(std::ostream& os, int repetitions){
  os << "index,kind,mean,stddev,median,";
  for (int i=0; i<repetitions-1; i++){
    os << i << ","; 
  }
  os << repetitions-1 << std::endl;
}
}

void movie_compress_bench_brighten(){
  bool time = true;
  auto copy = getCopyFunc();
  auto brighten = getBrightenFunc(20, true);
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j"), c("c");

  int repetitions = 100;

  auto bench_kind = getEnvVar("BENCH_KIND");
  Kind kind;
  if (bench_kind == "RLE"){
    kind = Kind::RLE;
    brighten = getBrightenFunc(20, false);
  } else if (bench_kind == "LZ77"){
    kind = Kind::LZ77;
  }

  auto start_str = getEnvVar("IMAGE_START");
  if (start_str == "") {
    std::cout << "No start" << std::endl;
    return;
  }
  auto end_str = getEnvVar("IMAGE_END");
  if (end_str == "") {
    std::cout << "No end" << std::endl;
    return;
  }

  int start = std::stoi(start_str);
  int end = std::stoi(end_str);

  auto folder1 = getEnvVar("FOLDER");
  if (folder1 == "") {
    std::cout << "No folder1" << std::endl;
    return;
  }

  std::string f1_temp = folder1;
  f1_temp.pop_back();
  auto found = f1_temp.find_last_of("/\\");
  f1_temp = f1_temp.substr(found+1);

  std::string name = "movie_recompress_brighten_" + to_string(kind) + "-" + f1_temp  + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;
  writeHeaderRecompress(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    auto frame_res = read_movie_frame(folder1, "frame", index, Kind::DENSE, w, h, f1_num_vals);
    auto dims = frame_res.first.getDimensions();

    auto frame = frame_res.first;

    Tensor<uint8_t> out("out", dims, {Dense,Dense,Dense});
    IndexStmt stmt = (out(i,j,c) = brighten(frame(i,j,c)));
    out.compile();
    out.assemble();
    out.compute();

    auto denseVals = out.getStorage().getValues();
    // std::vector<uint8_t> vals((uint8_t*)denseVals.getData(), (uint8_t*)denseVals.getData() + denseVals.getSize());
    int insize = denseVals.getSize();
    uint8_t* in  = (uint8_t*)denseVals.getData();
    std::cout << "in size: " << insize << std::endl;
    if (kind == Kind::LZ77){
      outputFile << index << "," << bench_kind << ",";
      uint8_t* out = 0;
      int outsize = 0;
      TOOL_BENCHMARK_REPEAT({
          encode_lz77_(in, insize, out, outsize);
          benchmark::DoNotOptimize(out);
      }, "Compute", repetitions, outputFile);
    }
  }
}