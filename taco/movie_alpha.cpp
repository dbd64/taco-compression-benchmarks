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

struct BlendOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Requires 2 arguments";
    return ir::Add::make(ir::Mul::make(0.7, v[0]), ir::Mul::make(0.3, v[1]));
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

Func Blend("blend", BlendOp(), unionAlgebra());
Func Blend_lz("blend_lz", BlendOp(), universeAlgebra());

void movie_alpha_bench(){
 bool time = true;
 auto copy = getCopyFunc();
 taco::util::TimeResults timevalue{};
 const IndexVar i("i"), j("j"), c("c");

 int repetitions = getNumRepetitions(100);

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

  auto folder1 = getEnvVar("FOLDER1");
  if (folder1 == "") {
    std::cout << "No folder1" << std::endl;
    return;
  }
  auto folder2 = getEnvVar("FOLDER2");
  if (folder2 == "") {
    std::cout << "No folder2" << std::endl;
    return;
  }

  std::string f1_temp = folder1;
  f1_temp.pop_back();
  auto found = f1_temp.find_last_of("/\\");
  f1_temp = f1_temp.substr(found+1);
  std::string f2_temp = folder2;
  f2_temp.pop_back();
  found = f2_temp.find_last_of("/\\");
  f2_temp = f2_temp.substr(found+1);

  std::string name = "movie_alpha_" + to_string(kind) + "-" + f1_temp + "-" + f2_temp + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;

  // std::ostream& outputFile = std::cout;
  writeHeader(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    std::cout << "movie_alpha: " << index <<  std::endl; 
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    int f2_num_vals = 0;
    auto f1 = read_movie_frame(folder1, "l", index, kind, w, h, f1_num_vals);
    auto f2 = read_movie_frame(folder2, "r", index, kind, w, h, f2_num_vals);
    auto dims = f1.first.getDimensions();

    auto f1t = f1.first;
    auto f2t = f2.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind == Kind::LZ77){
      stmt = (out(i) = Blend_lz(f1t(i), f2t(i)));
    } else {
      stmt = (out(i,j,c) = Blend(f1t(i,j,c),f2t(i,j,c)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = f1t.getStorage();
    taco_tensor_t* a2 = f2t.getStorage();

    outputFile << index << "," << bench_kind << "," << f1_num_vals + f2_num_vals << "," << f1.second + f2.second  << ",";
    TOOL_BENCHMARK_REPEAT({
        k.compute(a0,a1,a2);
    }, "Compute", repetitions, outputFile);

    out.compute();
    auto count = count_bytes_vals(out, kind);
    outputFile << "," << count.first << "," << count.second << std::endl;

    //  saveValidation(f1t, kind, w, h, false, bench_kind, index, "f1");
    //  saveValidation(f2t, kind, w, h, false,  bench_kind, index, "f2");
    // saveValidation(out, kind, w, h, false, bench_kind, index, "out");
  }
}