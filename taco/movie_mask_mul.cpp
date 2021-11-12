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

namespace {
struct MaskMulOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Requires 2 arguments (img1, mask)";
    return ir::Div::make(ir::Mul::make(v[0], v[1]), 255);
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

Func MaskMul_union("mask_union", MaskMulOp(), unionAlgebra());
Func MaskMul_universe("mask_universe", MaskMulOp(), universeAlgebra());
}

Tensor<uint8_t> gen_mask(Kind kind, int w, int h, int& maskBytes, int& maskVals, bool linearize){
  std::pair<Tensor<uint8_t>, int> p;
  double halfWidth = ((double)w)/2;
  if (linearize){
    vector<uint8_t> vals(w*h*3, 0);
    for (int r=0; r<h; r++){
      for (int c=w/2; c<w; c++){
        for (int color=0; color<3; color++){
          vals[r*w*3 + c*3 + color] = (255 * (c/halfWidth - 1));
        }
      }
    }
    p = to_vector_rgb(vals, h, w, 0, "mask_", kind, maskVals);
  } else {
    vector<uint8_t> vals(w*h, 0);
    for (int r=0; r<h; r++){
      for (int c=w/2; c<w; c++){
        vals[r*w + c] = (255 * (c/halfWidth - 1));
      }
    }
    p = to_tensor(vals, h, w, 0,"mask_", kind, maskVals);
  }
  maskBytes = p.second;
  return p.first;
}

void movie_mask_mul_bench(){
 bool time = true;
 auto copy = getCopyFunc();
 taco::util::TimeResults timevalue{};
 const IndexVar i("i"), j("j"), c("c");

 int repetitions = 100;

 auto bench_kind = getEnvVar("BENCH_KIND");
 Kind kind;
 Format f;
 if (bench_kind == "DENSE") {
   kind = Kind::DENSE;
   f = Format{Dense,Dense,Dense};
 } else if (bench_kind == "SPARSE"){
   kind = Kind::SPARSE;
   f = Format{Dense,Sparse,Dense};
 } else if (bench_kind == "RLE"){
   kind = Kind::RLE;
   f = Format{Dense,RLE_size(3),Dense};
 } else if (bench_kind == "LZ77"){
   kind = Kind::LZ77;
   f = Format{LZ77};
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

  std::string name = "movie_maskmul_" + to_string(kind) + "-" + f1_temp + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;

  // std::ostream& outputFile = std::cout;
  writeHeader(outputFile, repetitions);

  int w = 0;
  int h = 0;
  int f1_num_vals = 0;
  auto frame_res = read_movie_frame(folder1, "frame", start, kind, w, h, f1_num_vals);
  auto dims = frame_res.first.getDimensions();

  int maskBytes = 0;
  int maskVals = 0;
  auto mask = gen_mask(Kind::SPARSE, w, h, maskBytes, maskVals, kind == Kind::LZ77);
  // saveValidation(mask, kind, w, h,  bench_kind, 0, "mask", false);

  for (int index=start; index<=end; index++){
    std::cout << "movie_maskmul: " << index <<  std::endl; 

    frame_res = read_movie_frame(folder1, "frame", index, kind, w, h, f1_num_vals);
    auto frame = frame_res.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind == Kind::LZ77){
      stmt = (out(i) = MaskMul_universe(frame(i), mask(i)));
    } else {
      stmt = (out(i,j,c) = MaskMul_union(frame(i,j,c), mask(i,j)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = frame.getStorage();
    taco_tensor_t* a2 = mask.getStorage();

    outputFile << index << "," << bench_kind << "," << f1_num_vals + maskVals << "," << frame_res.second + maskBytes  << ",";
    TOOL_BENCHMARK_REPEAT({
        k.compute(a0,a1,a2);
    }, "Compute", repetitions, outputFile);

    out.compute();
    auto count = count_bytes_vals(out, kind);
    outputFile << "," << count.first << "," << count.second << std::endl;

    //  saveValidation(f1t, kind, w, h, false, bench_kind, index, "f1");
    // saveValidation(out, kind, w, h, false, bench_kind, index, "out");
  }
}