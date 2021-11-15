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

void movie_brighten_bench(){
 bool time = true;
 auto copy = getCopyFunc();
 auto brighten = getBrightenFunc(20, true);
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
   brighten = getBrightenFunc(20, false);
 } else if (bench_kind == "RLE"){
   kind = Kind::RLE;
   f = Format{Dense,RLE_size(3),Dense};
   brighten = getBrightenFunc(20, false);
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

  std::string name = "movie_brighten_" + to_string(kind) + "-" + f1_temp  + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;
  writeHeader(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    auto frame_res = read_movie_frame(folder1, "frame", index, kind, w, h, f1_num_vals);
    auto dims = frame_res.first.getDimensions();

    auto frame = frame_res.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind == Kind::LZ77){
      stmt = (out(i) = brighten(frame(i)));
    } else {
      stmt = (out(i,j,c) = brighten(frame(i,j,c)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = frame.getStorage();

    outputFile << index << "," << bench_kind << "," << frame_res.second << "," << f1_num_vals << ",";
    TOOL_BENCHMARK_REPEAT({
        k.compute(a0,a1);
    }, "Compute", repetitions, outputFile);

    out.compute();
    auto count = count_bytes_vals(out, kind);
    outputFile << "," << count.first << "," << count.second << std::endl;

    // saveValidation(f1t, kind, w, h, false, bench_kind, index, "f1");
    // saveValidation(f2t, kind, w, h, false, bench_kind, index, "f2");
    // if (kind == Kind::LZ77){
    //   saveValidation(mt,  kind, w, h, true, bench_kind, index, "mask");
    // } else {
    //   saveValidation(mt,  kind, w, h, bench_kind, index, "mask", true);
    // }
    // saveValidation(out, kind, w, h, false, bench_kind, index, "out");
  }
}