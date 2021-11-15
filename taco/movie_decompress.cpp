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

void movie_decompress_bench(){
 bool time = true;
 auto copy = getCopyFunc();
 taco::util::TimeResults timevalue{};
 const IndexVar i("i"), j("j"), c("c");

 int repetitions = getNumRepetitions(100);

 auto bench_kind = getEnvVar("BENCH_KIND");
 Kind kind;
 Format f;
 if (bench_kind == "RLE"){
   kind = Kind::RLE;
   f = Format{Dense,Dense,Dense};
 } else if (bench_kind == "LZ77"){
   kind = Kind::LZ77;
   f = Format{Dense};
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

  std::string name = "movie_decompress_dense_" + to_string(kind) + "-" + f1_temp + "-" + start_str + "-" + end_str + ".csv";
  name = getOutputPath() + name;
  std::ofstream outputFile(name);
  std::cout << "Starting " << name << std::endl;

  // std::ostream& outputFile = std::cout;
  writeHeader(outputFile, repetitions);

  for (int index=start; index<=end; index++){
    std::cout << "movie_decompress: " << index <<  std::endl; 
    int w = 0;
    int h = 0;
    int f1_num_vals = 0;
    int maskBytes = 0;
    int maskVals = 0;
    int imgBytes = 0;
    int imgVals = 0;

    auto frame_res = read_movie_frame(folder1, "frame", index, kind, w, h, f1_num_vals);
    auto dims = frame_res.first.getDimensions();

    auto frame = frame_res.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind == Kind::LZ77){
      stmt = (out(i) = copy(frame(i)));
    } else {
      stmt = (out(i,j,c) = copy(frame(i,j,c)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = frame.getStorage();

    outputFile << index << "," << bench_kind << "," << f1_num_vals + maskVals + imgVals << "," << frame_res.second + maskBytes + imgBytes << ",";
    TOOL_BENCHMARK_REPEAT({
        k.compute(a0,a1);
    }, "Compute", repetitions, outputFile);

    out.compute();
    auto count = count_bytes_vals(out, Kind::DENSE);
    outputFile << "," << count.first << "," << count.second << std::endl;

    // out.printComputeIR(std::cout);

    //  saveValidation(f1t, kind, w, h, false, bench_kind, index, "f1");
    //  saveValidation(f2t, kind, w, h, false,  bench_kind, index, "f2");
    // saveValidation(out, kind, w, h, false, bench_kind, index, "out");
  }
}