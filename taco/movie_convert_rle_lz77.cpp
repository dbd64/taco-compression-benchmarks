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

#define ALWAYS_INLINE __attribute__((always_inline))
// #define ALWAYS_INLINE 

using namespace taco;
namespace{
void writeHeaderRecompress(std::ostream& os, int repetitions){
  os << "index,kind,mean,stddev,median,";
  for (int i=0; i<repetitions-1; i++){
    os << i << ","; 
  }
  os << repetitions-1 << std::endl;
}

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

struct MaskOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 3) << "Requires 3 arguments (img1, img2, mask)";
    return ir::Add::make(ir::Mul::make(v[2], v[0]), ir::Mul::make(ir::Neg::make(ir::Cast::make(v[2], Bool)), v[1]));
  }
};

struct unionAlgebra {
 IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
   if (regions.size() == 2){
     return Union(regions[0], regions[1]);
   }
   auto t = Union(regions[0], regions[1]);
   for (int i=2; i< regions.size(); i++){
     t = Union(t, regions[i]);
   }
   return t;
 }
};

struct universeAlgebra {
 IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
   auto t = Union(regions[0], Complement(regions[0]));
   for (int i=1; i< regions.size(); i++){
     t = Union(t, Union(regions[i], Complement(regions[i])));
   }
   return t;
 }
};

Func Mask("mask", MaskOp(), unionAlgebra());
Func Mask_lz("mask_lz", MaskOp(), universeAlgebra());

void lz77_to_rle(const uint8_t* in, const int insize, uint8_t*& out, int& outsize, int*& crd, int& crdsize){
  int outcap = 1048576;
  out = (uint8_t*) malloc(outcap);
  outsize = 0;

  int crdcap = 1048576;
  crd = (int*) malloc(crdcap*sizeof(int));
  crdsize = 0;

  int curr_crd = 0;

  size_t i = 0;
  while (i<insize){
    if ((in[i+1] >> 7 & 1) == 0){
      uint16_t numBytes = *((uint16_t *) &in[i]);
      i+=2;
      if (outsize+numBytes >= outcap-1){
        outcap = (outsize+numBytes)*2;
        out = (uint8_t*)realloc((void*)out, outcap);
      }
      if (crdsize+numBytes >= crdcap-1){
        crdcap = (crdsize+numBytes)*2;
        crd = (int*)realloc((void*)crd, crdcap*sizeof(int));
      }
      for(int j=0; j<numBytes; j++){
        out[outsize++] = in[i+j];
        crd[crdsize++] = curr_crd++;
      }
      i+=numBytes;
    } else  {
      uint16_t dist = *((uint16_t *) &in[i+2]);
      uint16_t run = *((uint16_t *) &in[i]) & (uint16_t)0x7FFF;
      if (dist == 3 && run > 4){
        int curr_pos = i - dist;
        int num_remaining = run;
        auto numToPush = curr_crd % 3;
        if (numToPush != 0){
          num_remaining-= numToPush;
          if (outsize+numToPush >= outcap-1){
            outcap = (outsize+numToPush)*2;
            out = (uint8_t*)realloc((void*)out, outcap);
          }
          if (crdsize+numToPush >= crdcap-1){
            crdcap = (crdsize+numToPush)*2;
            crd = (int*)realloc((void*)crd, crdcap*sizeof(int));
          }
          while (curr_crd % 3 != 0){
            out[outsize++] = in[curr_pos++];
            crd[crdsize++] = curr_crd++;
          }
        }
        int start_save = outsize - 3;
        // Updating the coordinate is what sets the run length
        curr_crd += num_remaining/3;
        num_remaining -= num_remaining/3;
        if (num_remaining != 0){
          if (outsize+num_remaining >= outcap-1){
            outcap = (outsize+num_remaining)*2;
            out = (uint8_t*)realloc((void*)out, outcap);
          }
          if (crdsize+num_remaining >= crdcap-1){
            crdcap = (crdsize+num_remaining)*2;
            crd = (int*)realloc((void*)crd, crdcap*sizeof(int));
          }
          while (num_remaining != 0){
            out[outsize++] = out[start_save++];
            crd[crdsize++] = curr_crd++;
            num_remaining--;
          }
        }
      } else if (dist == 1 && run > 4){
                int curr_pos = i - dist;
        int num_remaining = run;
        auto numToPush = curr_crd % 3;
        if (numToPush != 0){
          num_remaining-= numToPush;
          if (outsize+numToPush >= outcap-1){
            outcap = (outsize+numToPush)*2;
            out = (uint8_t*)realloc((void*)out, outcap);
          }
          if (crdsize+numToPush >= crdcap-1){
            crdcap = (crdsize+numToPush)*2;
            crd = (int*)realloc((void*)crd, crdcap*sizeof(int));
          }
          while (curr_crd % 3 != 0){
            out[outsize++] = in[curr_pos++];
            crd[crdsize++] = curr_crd++;
          }
        }
        // Updating the coordinate is what sets the run length
        curr_crd += num_remaining;
        num_remaining -= num_remaining/3;
      } else {
        // std::cout << "dist,run: " << dist << ", " << run << std::endl;
        if (outsize+run >= outcap-1){
          outcap = (outsize+run)*2;
          out = (uint8_t*)realloc((void*)out, outcap);
        }
        if (crdsize+run >= crdcap-1){
          crdcap = (crdsize+run)*2;
          crd = (int*)realloc((void*)crd, crdcap*sizeof(int));
        }
        if (dist <= run){
          for (size_t j = i-dist; j<i; j++){
            out[outsize++] = in[j];
            crd[crdsize++] = curr_crd++;
          }
          size_t start = outsize-dist;
          for (size_t j = 0; j<(run-dist); j++){
            out[outsize++] = out[start+j];
            crd[crdsize++] = curr_crd++;
          }
        } else {
          for (size_t j = i-dist; j < i-dist+run; j++){
            out[outsize++] = in[j];
            crd[crdsize++] = curr_crd++;
          }
        }
      }
      i+=4;
    }
  }
}

ALWAYS_INLINE
void set_uint16(uint8_t*& out, int& outsize, int& outcap, const int index, uint16_t value){
  uint8_t bytes[sizeof(value)];
  memcpy(bytes, &value, sizeof(value));

  out[index + 0] = bytes[0];
  out[index + 1] = bytes[1];
}

ALWAYS_INLINE
void next_defined(int&i, int &j, int& jbPos, int* __restrict__ B2_pos, int* __restrict__ B2_crd, int B1_dimension){
  if (++jbPos < B2_pos[i+1]){
    j = B2_crd[jbPos];
  } else {
    i++;
    while(i < B1_dimension && B2_pos[i] == B2_pos[i+1] ){
      i++;
    }
    if (i == B1_dimension){
      j = 0;
    } else {
      jbPos = B2_pos[i];
      j = B2_crd[jbPos];
    }
  }
}

ALWAYS_INLINE
void push_defined_value(uint8_t*& out, int& outsize, int& outcap, int& jbPos, const int i, const int j, const int B2_dimension, uint8_t* __restrict__ B_vals){
  if (outsize+5 >= outcap-1){
    outcap = (outsize+3)*2;
    out = (uint8_t*)realloc((void*)out, outcap);
  }
  set_uint16(out, outsize, outcap, outsize, 3); // Set the count to three
  int32_t pos = 3*(jbPos);
  for (int c=0; c<3; c++){
    out[outsize++] = B_vals[pos++];
  }
}

void rle_to_lz77(taco_tensor_t *B, uint8_t*& out, int& outsize){
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  int B3_dimension = (int)(B->dimensions[2]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  int* __restrict__ B2_crd = (int*)(B->indices[1][1]);
  uint8_t* __restrict__ B_vals = (uint8_t*)(B->vals);
  uint8_t B_fill_value = *((uint8_t*)(B->fill_value));
  uint8_t* B_fill_region = ((uint8_t*)(B->fill_value));
  int32_t B_fill_len = 1;
  int32_t B_fill_index = 0;

  int count_idx = -1;
  int count = 0;
  int outcap = 1048576;
  out = (uint8_t*) malloc(outcap);
  outsize = 0;

  int i = 0;
  int j = 0;
  int jbPos = 0;
  while (i*B2_dimension + j < B1_dimension * B2_dimension){
    int curr_i = i;
    int curr_j = j;
    push_defined_value(out, outsize, outcap, jbPos, i, j, B2_dimension, B_vals);
    next_defined(i, j, jbPos, B2_pos, B2_crd, B1_dimension);
    int run = ((i*B2_dimension + j) - (curr_i*B2_dimension + curr_j + 1))*3;
    while (run){
      int curr_run = run;
      if (run > 32767){
        curr_run = 32767;
      }
      run -= curr_run;
      if (outsize+4 >= outcap-1){
        outcap = (outsize+4)*2;
        out = (uint8_t*)realloc((void*)out, outcap);
      }
      set_uint16(out, outsize, outcap, outsize, 32768 | curr_run);
      set_uint16(out, outsize, outcap, outsize+2, 3);
      outsize+=4;
      if (run){
        push_defined_value(out, outsize, outcap, jbPos, curr_i, curr_j, B2_dimension, B_vals);
        run-=3;
      }
    }
  }
}
}

void movie_lz77_rle(){
 bool time = true;
 auto copy = getCopyFunc();
 taco::util::TimeResults timevalue{};
 const IndexVar i("i"), j("j"), c("c");

 int repetitions = getNumRepetitions(100);

 auto bench_kind = getEnvVar("BENCH_KIND");
 Kind kind_from;
 Kind kind_to;
 Format f;
 if (bench_kind == "RLE"){
   kind_from = Kind::RLE;
   kind_to = Kind::LZ77;
   f = {Dense, RLE_size(3), Dense};
 } else if (bench_kind == "LZ77"){
   kind_from = Kind::LZ77;
   kind_to = Kind::RLE;
   f = {LZ77};
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
  int numFrames = end - (start-1);

  auto folder1 = getEnvVar("FOLDER");
  if (folder1 == "") {
    std::cout << "No folder1" << std::endl;
    return;
  }

  std::string f1_temp = folder1;
  f1_temp.pop_back();
  auto found = f1_temp.find_last_of("/\\");
  f1_temp = f1_temp.substr(found+1);

  std::string name = "movie_rlelz77_" + to_string(kind_from) + "-" + f1_temp + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;

  // std::ostream& outputFile = std::cout;
  writeHeader(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    std::cout << "rle_lz77: " << index <<  std::endl; 
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    int maskBytes = 0;
    int maskVals = 0;
    int imgBytes = 0;
    int imgVals = 0;

    auto frame_res = read_movie_frame(folder1, "frame", index, kind_from, w, h, f1_num_vals);
    auto dims = frame_res.first.getDimensions();
    auto frame = frame_res.first;


    outputFile << index << "," << bench_kind << "," << f1_num_vals + maskVals + imgVals << "," << frame_res.second + maskBytes + imgBytes << ",";
    if (kind_from == Kind::RLE){
      taco_tensor_t* out_tensor = frame.getStorage(); 
      uint8_t* out = 0;
      int outsize = 0;
      TOOL_BENCHMARK_REPEAT({
          rle_to_lz77(out_tensor, out, outsize);
          benchmark::DoNotOptimize(out);
          benchmark::DoNotOptimize(outsize);
      }, "Compute", repetitions, outputFile);
    } else {
      auto denseVals = frame.getStorage().getValues();
      int insize = denseVals.getSize();
      uint8_t* in  = (uint8_t*)denseVals.getData();
      uint8_t* out = 0;
      int outsize = 0;
      int* crd = 0;
      int crdsize = 0;
      TOOL_BENCHMARK_REPEAT({
        lz77_to_rle(in, insize, out, outsize, crd, crdsize);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(outsize);
        benchmark::DoNotOptimize(crd);
        benchmark::DoNotOptimize(crdsize);
      }, "Compute", repetitions, outputFile);
    }

    // out.compute();
    // auto count = count_bytes_vals(out, Kind::DENSE);
    // outputFile << "," << count.first << "," << count.second << std::endl;
    outputFile << std::endl;
  }
}

void movie_rle_lz77_bench_brighten(){
  bool time = true;
  auto copy = getCopyFunc();
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j"), c("c");

  int repetitions = getNumRepetitions(100);

  auto bench_kind = getEnvVar("BENCH_KIND");
  Kind kind_from;
  Kind kind_to;
  Format f;
  if (bench_kind == "RLE"){
    kind_from = Kind::RLE;
    kind_to = Kind::LZ77;
    f = {Dense, RLE_size(3), Dense};
  } else if (bench_kind == "LZ77"){
    kind_from = Kind::LZ77;
    kind_to = Kind::RLE;
    f = {LZ77};
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

  std::string name = "movie_lz77_rle_brighten_" + to_string(kind_from) + "-" + f1_temp  + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;
  writeHeaderRecompress(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    auto frame_res = read_movie_frame(folder1, "frame", index, kind_from, w, h, f1_num_vals);
    auto dims = frame_res.first.getDimensions();

    auto frame = frame_res.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind_from == Kind::LZ77){
      auto brighten = getBrightenFunc(20, true);
      stmt = (out(i) = brighten(frame(i)));
    } else {
      auto brighten = getBrightenFunc(20, false);
      stmt = (out(i,j,c) = brighten(frame(i,j,c)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    out.compute();

    std::cout << "index " << index << std::endl;
    outputFile << index << "," << bench_kind << ",";
    if (kind_from == Kind::RLE){
      taco_tensor_t* out_tensor = out.getStorage(); 
      uint8_t* out = 0;
      int outsize = 0;
      TOOL_BENCHMARK_REPEAT({
          rle_to_lz77(out_tensor, out, outsize);
          benchmark::DoNotOptimize(out);
          benchmark::DoNotOptimize(outsize);
      }, "Compute", repetitions, outputFile);
    } else {
      auto denseVals = out.getStorage().getValues();
      int insize = denseVals.getSize();
      uint8_t* in  = (uint8_t*)denseVals.getData();
      uint8_t* out = 0;
      int outsize = 0;
      int* crd = 0;
      int crdsize = 0;
      TOOL_BENCHMARK_REPEAT({
        lz77_to_rle(in, insize, out, outsize, crd, crdsize);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(outsize);
        benchmark::DoNotOptimize(crd);
        benchmark::DoNotOptimize(crdsize);
      }, "Compute", repetitions, outputFile);
    }
    outputFile << std::endl;
  }
}

void movie_rle_lz77_bench_subtitle(){
  bool time = true;
  auto copy = getCopyFunc();
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j"), c("c");

  int repetitions = getNumRepetitions(100);

  auto bench_kind = getEnvVar("BENCH_KIND");
  Kind kind_from;
  Kind kind_to;
  Format f;
  if (bench_kind == "RLE"){
    kind_from = Kind::RLE;
    kind_to = Kind::LZ77;
    f = {Dense, RLE_size(3), Dense};
  } else if (bench_kind == "LZ77"){
    kind_from = Kind::LZ77;
    kind_to = Kind::RLE;
    f = {LZ77};
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

  std::string name = "movie_rle_lz77_subtitle_" + to_string(kind_from) + "-" + f1_temp  + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;
  writeHeaderRecompress(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    int maskBytes = 0;
    int maskVals = 0;
    int imgBytes = 0;
    int imgVals = 0;

    auto frame_res = read_movie_frame(folder1, "frame", index, kind_from, w, h, f1_num_vals);
    auto mask_res = read_subtitle_mask(kind_from, w, h, maskBytes, maskVals, imgBytes, imgVals);
    auto dims = frame_res.first.getDimensions();

    auto frame = frame_res.first;
    auto subtitle = mask_res.first;
    auto mask = mask_res.second;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind_from == Kind::LZ77){
      stmt = (out(i) = Mask_lz(frame(i), subtitle(i), mask(i)));
    } else {
      stmt = (out(i,j,c) = Mask(frame(i,j,c), subtitle(i,j), mask(i,j)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    out.compute();

    std::cout << "index " << index << std::endl;
    outputFile << index << "," << bench_kind << ",";
    if (kind_from == Kind::RLE){
      taco_tensor_t* out_tensor = out.getStorage(); 
      uint8_t* out = 0;
      int outsize = 0;
      TOOL_BENCHMARK_REPEAT({
          rle_to_lz77(out_tensor, out, outsize);
          benchmark::DoNotOptimize(out);
          benchmark::DoNotOptimize(outsize);
      }, "Compute", repetitions, outputFile);
    } else {
      auto denseVals = out.getStorage().getValues();
      int insize = denseVals.getSize();
      uint8_t* in  = (uint8_t*)denseVals.getData();
      uint8_t* out = 0;
      int outsize = 0;
      int* crd = 0;
      int crdsize = 0;
      TOOL_BENCHMARK_REPEAT({
        lz77_to_rle(in, insize, out, outsize, crd, crdsize);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(outsize);
        benchmark::DoNotOptimize(crd);
        benchmark::DoNotOptimize(crdsize);
      }, "Compute", repetitions, outputFile);
    }
    outputFile << std::endl;
  }
}


