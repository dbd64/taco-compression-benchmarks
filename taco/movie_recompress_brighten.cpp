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
int computeHash(const std::vector<uint8_t>& in, int in_idx){
  return ( ( in[ in_idx ] & 0xFF ) << 8 ) |
         ( ( in[ in_idx + 1 ] & 0xFF ) ^ ( in[ in_idx + 2 ] & 0xFF ) );
}

std::pair<int,int> find_match(const int in_idx, const int out_idx, const int start_idx,
                              const std::vector<uint8_t>& in, const std::vector<uint8_t>& out) {
  int len = 1;
  int off = out_idx - start_idx;
  if( off > 0 && off < 65536 && start_idx >= 0 && start_idx < out.size() &&
      out[ start_idx ] == in[ in_idx ] ) {
    while( in_idx + len < in.size() && len <= MAX_RUN && 
           out[ start_idx + (len % off) ] == in[ in_idx + len ] ) {
      len++;
    }
  }
  return {len, off};
}

union short_byte {
    uint16_t val;
    uint8_t bytes[2];
};

void push_uint16(std::vector<uint8_t>& out, uint16_t value){
  short_byte store{value};
  out.push_back(store.bytes[0]);
  out.push_back(store.bytes[1]);
}

void set_uint16(std::vector<uint8_t>& out, int index, uint16_t value){
  short_byte store{value};
  out[index + 0] = store.bytes[0];
  out[index + 1] = store.bytes[1];
}

uint16_t load_uint16(std::vector<uint8_t>& out, int index){
  short_byte store;
  store.bytes[0] = out[index];
  store.bytes[1] = out[index+1];
  return store.val;
}

void encode_lz77_(const std::vector<uint8_t>& in, std::vector<uint8_t> out) {
  int in_idx = 0;
  int count_idx = -1;
  std::vector<int> hash(65536);

  while( in_idx < in.size() ) {
    int len = 1;
    int off = 0;
    if( in_idx + 2 < in.size() ) {
      auto rle = find_match(in_idx, out.size(), out.size() - 1, in, out);
      auto pixelRle = find_match(in_idx, out.size(), out.size() - 3, in, out);
      auto hashCheck = find_match(in_idx, out.size(), hash[ computeHash(in, in_idx) ], in, out);
      len = std::max({rle.first, pixelRle.first, hashCheck.first});
      if (rle.first == len){
        off = rle.second;
      } else if (pixelRle.first == len) {
        off = pixelRle.second;
      } else {
        off = hashCheck.second;
      }
    }
    if( len >= 4 ) {
      // if( len > MAX_LIT ) {
      //   len = MAX_LIT;
      // }
      push_uint16(out, 32768 | len);
      push_uint16(out, off);
      count_idx = -1;
      hash[computeHash(out, out.size()-4)] = out.size()-4;
      hash[computeHash(out, out.size()-3)] = out.size()-3;
      in_idx+=len;
    } else {
      while( len ) {
        if (count_idx != -1 && load_uint16(out, count_idx)==MAX_LIT){
          count_idx = -1;
        }
        if (count_idx == -1) {
          count_idx = out.size();
          push_uint16(out, 0);
        }
        
        int old_cout = load_uint16(out, count_idx);

        int new_count = len+old_cout > MAX_LIT ? MAX_LIT : len+old_cout;
        int max = new_count - old_cout;
        len -= max;
        set_uint16(out, count_idx, new_count);
        if (count_idx+2<out.size()) {
          hash[computeHash(out, count_idx)] = count_idx;
        }
        while( max ) {
          out.push_back(in[ in_idx++ ]);
          if(out.size() > 2) {
            hash[computeHash(out, out.size()-3)] = out.size()-3;
          }
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
    std::vector<uint8_t> vals((uint8_t*)denseVals.getData(), (uint8_t*)denseVals.getData() + denseVals.getSize());
    if (kind == Kind::LZ77){
      outputFile << index << "," << bench_kind << ",";
      std::vector<uint8_t> reencoded;
      reencoded.reserve(1048576);
      TOOL_BENCHMARK_REPEAT({
          encode_lz77_(vals, reencoded);
          benchmark::DoNotOptimize(reencoded);
      }, "Compute", repetitions, outputFile);
    }
  }
}