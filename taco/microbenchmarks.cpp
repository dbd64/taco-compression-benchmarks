#include "bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"
#include "taco/util/timers.h"
#include "utils.h"
#include "rapidcsv.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"
#include "png_reader.h"

#include <random>

using namespace taco;

// mktemp -d  $(pwd)/build_dirs/tmp.XXXXXXXXXX

namespace {
bool useRLEVector = true;

struct ConstMulOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 1) << "Requires 1 arguments";
    return ir::Mul::make(v[0], 7);
  }
};

struct MulOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Requires 2 arguments";
    return ir::Mul::make(v[0], v[1]);
  }
};

struct MaskMulOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Requires 2 arguments";
    return ir::Div::make(ir::Mul::make(v[0], v[1]), 255);
  }
};

struct unionAlgebra {
 IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    if (regions.size() == 1){
        return regions[0];
    } else if (regions.size() == 2){
      return Union(regions[0], regions[1]);
    }
    auto t = Union(regions[0], regions[1]);
    for (int i=2; i< regions.size(); i++){
      t = Union(t, regions[i]);
    }
    return t;
 }
};

struct intersectionAlgebra {
 IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
   if (regions.size() == 2){
     return Intersect(regions[0], regions[1]);
   }
   auto t = Intersect(regions[0], regions[1]);
   for (int i=2; i< regions.size(); i++){
     t = Intersect(t, regions[i]);
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

Func ConstMul_intersect("mul_intersect", ConstMulOp(), intersectionAlgebra());
Func ConstMul_union("mul_union", ConstMulOp(), unionAlgebra());
Func ConstMul_universe("mul_universe", ConstMulOp(), universeAlgebra());

Func Mul_intersect("mul_intersect", MulOp(), intersectionAlgebra());
Func Mul_union("mul_union", MulOp(), unionAlgebra());
Func Mul_universe("mul_universe", MulOp(), universeAlgebra());

Func MaskMul_intersect("mul_intersect", MaskMulOp(), intersectionAlgebra());
Func MaskMul_union("mul_union", MaskMulOp(), unionAlgebra());
Func MaskMul_universe("mul_universe", MaskMulOp(), universeAlgebra());


Tensor<int> gen_mask(Kind kind, int w, int h, int& maskBytes, int& maskVals, bool linearize){
  std::pair<Tensor<int>, int> p;
//   double halfWidth = w - ((double)w)/8;
  int start_col = w - (w/8);
  if (linearize){
    // vector<int> vals(w*h*3, 0);
    Tensor<int> t{"mask_sparse", {h*w}, {Sparse}, 0};
    for (int r=0; r<h; r++){
      for (int c=start_col; c<w; c++){
        t(r*w + c) = 1; //(255 * (c/halfWidth - 1));
      }
    }
    t.pack();
    maskVals = t.getStorage().getValues().getSize();
    p = {t, t.getStorage().getValues().getSize()*2};
  } else 
  {
    vector<int> vals(w*h, 0);
    for (int r=0; r<h; r++){
      for (int c=start_col; c<w; c++){
        vals[r*w + c] = 1; // (255 * (c/halfWidth - 1));
      }
    }
    p = to_tensor_int(vals, h, w, 0,"mask_", kind, maskVals);
  }
  maskBytes = p.second;
  return p.first;
}


Tensor<int> makeRLEVector(std::string name, std::vector<int> v, int size){
    Tensor<int> t{name, {size}, {RLE}, 0};
    int curr = v[0];
    t(0) = curr;
    for (int i=0; i<size; i++){
        if (v[i] != curr){
            curr = v[i];
            t(i) = curr;
        }
    }
    t.pack();
    return t;
}

std::pair<Tensor<int>, Tensor<int>> gen_rand(int index, int width, int height, int run_upper, Kind kind, int& numVals, int& numBytes){
    std::default_random_engine gen(index);
    std::uniform_int_distribution<int> unif_vals(0, 255);
    std::uniform_int_distribution<int> unif_runs(1, run_upper);

    std::vector<int> v;
    int label = unif_vals(gen);
    int run_len = unif_runs(gen);
    int numValsVec = 1;
    for (int i=0; i<width; i++){
        if (run_len == 0 ){
            label = unif_vals(gen);
            run_len = unif_runs(gen);
            numValsVec++;
        }
        v.push_back(label);
        run_len--;
    }
    Tensor<int> vec = useRLEVector ? makeRLEVector("vec_rand", v, v.size()) : makeDenseVector("vec_rand", v.size(), v);

    // Load matrix
    if (kind == Kind::LZ77){
        std::vector<TempValue<int>> m;
        label = unif_vals(gen);
        run_len = unif_runs(gen);
        int remaining = width*height;
        while (remaining > 0){
            m.push_back(label);
            run_len-=1; remaining -= 1; numVals++;
            int count = std::min({run_len, 32767, remaining});
            if (count) m.push_back(Repeat{1,count});
            run_len -= count;
            remaining -= count;
            if (run_len == 0){
                label = unif_vals(gen);
                run_len = unif_runs(gen);
            }
        }
        auto packed = packLZ77_bytes(m);
        numBytes = packed.first.size() + 2*packed.second.size();
        numVals += numValsVec;
        return {vec, makeLZ77<int>("mtx_rand_" + std::to_string(index), {width*height}, 
                                   {0, (int) packed.first.size()}, packed.first, 
                                   {0, (int) packed.second.size()}, packed.second)};
    } else {
        std::vector<int> m;
        label = unif_vals(gen);
        run_len = unif_runs(gen);
        for (int i=0; i<width*height; i++){
            if (run_len == 0 ){
                label = unif_vals(gen);
                run_len = unif_runs(gen);
            }
            m.push_back(label);
            run_len--;
        }

        auto mat = to_tensor_int(m, height, width, 0, "mtx_rand_" + std::to_string(index), kind, numVals, 0);
        numBytes = mat.second;
        // numVals += numValsVec;
        return {vec, mat.first};
    }
}

void writeHeaderSpmvRand(std::ostream& os, int repetitions){
  os << "index,kind,width,height,run_upper,total_vals,total_bytes,mean,stddev,median,";
  for (int i=0; i<repetitions-1; i++){
    os << i << ","; 
  }
  os << repetitions-1 << ",";
  os << "out_bytes,out_vals";
  os << std::endl;
}


}

void bench_constmul_rand(){
    bool time = true;
    auto copy = getCopyFunc();
    taco::util::TimeResults timevalue{};
    const IndexVar i("i"), j("j"), c("c");

    int repetitions = getNumRepetitions(100);

    auto cache_str = getEnvVar("CACHE");
    bool cold_cache = true;
    if (cache_str == "WARM"){
        cold_cache = false;
    } else {
        cache_str = "COLD";
    }

    Func func = ConstMul_intersect;

    auto bench_kind = getEnvVar("BENCH_KIND");
    Kind kind;
    Format f;
    if (bench_kind == "DENSE") {
        kind = Kind::DENSE;
        f = Format{Dense,Dense};
        func = ConstMul_universe;
    } else if (bench_kind == "SPARSE"){
        kind = Kind::SPARSE;
        f = Format{Dense,Sparse};
        func = ConstMul_union;
    } else if (bench_kind == "RLE"){
        kind = Kind::RLE;
        f = Format{Dense,RLE};
        func = ConstMul_union;
    } else if (bench_kind == "LZ77"){
        kind = Kind::LZ77;
        f = Format{LZ77};
        func = ConstMul_universe;
    }


    int width = getIntEnvVar("RAND_WIDTH", 1000);
    int height = getIntEnvVar("RAND_HEIGHT", 10000);
    auto run_upper_str = getEnvVar("RUN_UPPER");
    int run_upper = std::stoi(run_upper_str);

    auto index_str = getEnvVar("INDEX");
    int index = std::stoi(index_str);


    std::string name = std::string(useRLEVector ? "RLE_" : "DENSE_") + "constmul_" + cache_str + "_" + bench_kind + "_" + index_str + "_" + run_upper_str + "_RAND.csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeaderSpmvRand(outputFile, repetitions);

    // for (int index=start; index<=end; index++)
    {
        int numVals = 0;
        int numBytes = 0;
        auto ins = gen_rand(index, width, height, run_upper, kind, numVals, numBytes);
        Tensor<int> matrix = ins.second;

        std::cout << matrix.getDimensions() << std::endl;

        Tensor<uint8_t> out("out", matrix.getDimensions(), f);
        IndexStmt stmt;
        if (kind == Kind::LZ77){
            stmt = (out(i) = func(matrix(i)));
        } else {
            stmt = (out(i,j) = func(matrix(i,j)));
        }

        out.setAssembleWhileCompute(true);
        out.compile();
        Kernel k = getKernel(stmt, out);

        taco_tensor_t* a0 = out.getStorage();
        taco_tensor_t* a1 = matrix.getStorage();

        outputFile << index << "," << bench_kind << "," << width << "," << height << "," << run_upper << "," << numVals << "," << numBytes  << ",";
        if (cold_cache){
            TOOL_BENCHMARK_REPEAT({
                k.compute(a0,a1);
            }, "Compute", repetitions, outputFile);
        } else {
            TOOL_BENCHMARK_REPEAT_WARM({
                k.compute(a0,a1);
            }, "Compute", repetitions, outputFile);
        }

        out.compute();
        auto count = count_bytes_vals(out, kind);
        outputFile << "," << count.first << "," << count.second << std::endl;

        out.printComputeIR(std::cout);
    }

}

void bench_elementwisemul_rand(){
    bool time = true;
    auto copy = getCopyFunc();
    taco::util::TimeResults timevalue{};
    const IndexVar i("i"), j("j"), c("c");

    int repetitions = getNumRepetitions(100);

    auto cache_str = getEnvVar("CACHE");
    bool cold_cache = true;
    if (cache_str == "WARM"){
        cold_cache = false;
    } else {
        cache_str = "COLD";
    }

    Func func = Mul_intersect;

    auto bench_kind = getEnvVar("BENCH_KIND");
    Kind kind;
    Format f;
    if (bench_kind == "DENSE") {
        kind = Kind::DENSE;
        f = Format{Dense,Dense};
        func = Mul_universe;
    } else if (bench_kind == "SPARSE"){
        kind = Kind::SPARSE;
        f = Format{Dense,Sparse};
        func = Mul_intersect;
    } else if (bench_kind == "RLE"){
        kind = Kind::RLE;
        f = Format{Dense,RLE};
        func = Mul_union;
    } else if (bench_kind == "LZ77"){
        kind = Kind::LZ77;
        f = Format{LZ77};
        func = Mul_universe;
    }


    int width = getIntEnvVar("RAND_WIDTH", 1000);
    int height = getIntEnvVar("RAND_HEIGHT", 10000);
    auto run_upper_str = getEnvVar("RUN_UPPER");
    int run_upper = std::stoi(run_upper_str);

    auto index_str = getEnvVar("INDEX");
    int index = std::stoi(index_str);


    std::string name = std::string(useRLEVector ? "RLE_" : "DENSE_") + "elementwisemul_" + cache_str + "_" + bench_kind + "_" + index_str + "_" + run_upper_str + "_RAND.csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeaderSpmvRand(outputFile, repetitions);

    // for (int index=start; index<=end; index++)
    {
        int numVals0 = 0;
        int numBytes0 = 0;
        auto ins = gen_rand(index, width, height, run_upper, kind, numVals0, numBytes0);
        Tensor<int> matrix = ins.second;

        int numVals1 = 0;
        int numBytes1 = 0;
        auto ins1 = gen_rand(index+10, width, height, run_upper, kind, numVals1, numBytes1);
        Tensor<int> matrix1 = ins1.second;

        std::cout << matrix.getDimensions() << std::endl;

        Tensor<uint8_t> out("out", matrix.getDimensions(), f);
        IndexStmt stmt;
        if (kind == Kind::LZ77){
            stmt = (out(i) = func(matrix(i), matrix1(i)));
        } else {
            stmt = (out(i,j) = func(matrix(i,j), matrix1(i,j)));
        }

        out.setAssembleWhileCompute(true);
        out.compile();
        Kernel k = getKernel(stmt, out);

        taco_tensor_t* a0 = out.getStorage();
        taco_tensor_t* a1 = matrix.getStorage();
        taco_tensor_t* a2 = matrix1.getStorage();

        outputFile << index << "," << bench_kind << "," << width << "," << height << "," << run_upper << "," << numVals0 + numVals1 << "," << numBytes0 + numBytes1  << ",";
        if (cold_cache){
            TOOL_BENCHMARK_REPEAT({
                k.compute(a0,a1,a2);
            }, "Compute", repetitions, outputFile);
        } else {
            TOOL_BENCHMARK_REPEAT_WARM({
                k.compute(a0,a1,a2);
            }, "Compute", repetitions, outputFile);
        }

        out.compute();
        auto count = count_bytes_vals(out, kind);
        outputFile << "," << count.first << "," << count.second << std::endl;

        out.printComputeIR(std::cout);
    }

}

void bench_maskmul_rand(){
    bool time = true;
    auto copy = getCopyFunc();
    taco::util::TimeResults timevalue{};
    const IndexVar i("i"), j("j"), c("c");

    int repetitions = getNumRepetitions(100);

    auto cache_str = getEnvVar("CACHE");
    bool cold_cache = true;
    if (cache_str == "WARM"){
        cold_cache = false;
    } else {
        cache_str = "COLD";
    }

    Func func = MaskMul_intersect;

    auto bench_kind = getEnvVar("BENCH_KIND");
    Kind kind;
    Format f;
    if (bench_kind == "DENSE") {
        kind = Kind::DENSE;
        f = Format{Dense,Dense};
        func = MaskMul_universe;
    } else if (bench_kind == "SPARSE"){
        kind = Kind::SPARSE;
        f = Format{Dense,Sparse};
        func = MaskMul_intersect;
    } else if (bench_kind == "RLE"){
        kind = Kind::RLE;
        f = Format{Dense,RLE};
        func = MaskMul_universe;
    } else if (bench_kind == "LZ77"){
        kind = Kind::LZ77;
        f = Format{LZ77};
        func = MaskMul_universe;
    }


    int width = getIntEnvVar("RAND_WIDTH", 1000);
    int height = getIntEnvVar("RAND_HEIGHT", 10000);
    auto run_upper_str = getEnvVar("RUN_UPPER");
    int run_upper = std::stoi(run_upper_str);

    auto index_str = getEnvVar("INDEX");
    int index = std::stoi(index_str);


    std::string name = std::string(useRLEVector ? "RLE_" : "DENSE_") + "maskmul_" + cache_str + "_" + bench_kind + "_" + index_str + "_" + run_upper_str + "_RAND.csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeaderSpmvRand(outputFile, repetitions);

    // for (int index=start; index<=end; index++)
    {
        int numVals = 0;
        int numBytes = 0;
        auto ins = gen_rand(index, width, height, run_upper, kind, numVals, numBytes);
        Tensor<int> matrix = ins.second;

        int maskVals = 0;
        int maskBytes = 0;
        auto matrix1 = gen_mask(Kind::SPARSE, width, height, maskBytes, maskVals, kind == Kind::LZ77); //gen_rand(index+10, width, height, run_upper, kind, numVals, numBytes);

        std::cout << matrix.getDimensions() << std::endl;

        Tensor<uint8_t> out("out", matrix.getDimensions(), f);
        IndexStmt stmt;
        if (kind == Kind::LZ77){
            stmt = (out(i) = func(matrix(i), matrix1(i)));
        } else {
            stmt = (out(i,j) = func(matrix(i,j), matrix1(i,j)));
        }

        out.setAssembleWhileCompute(true);
        out.compile();
        Kernel k = getKernel(stmt, out);

        taco_tensor_t* a0 = out.getStorage();
        taco_tensor_t* a1 = matrix.getStorage();
        taco_tensor_t* a2 = matrix1.getStorage();

        outputFile << index << "," << bench_kind << "," << width << "," << height << "," << run_upper << "," << numVals + maskVals << "," << numBytes + maskBytes  << ",";
        if (cold_cache){
            TOOL_BENCHMARK_REPEAT({
                k.compute(a0,a1,a2);
            }, "Compute", repetitions, outputFile);
        } else {
            TOOL_BENCHMARK_REPEAT_WARM({
                k.compute(a0,a1,a2);
            }, "Compute", repetitions, outputFile);
        }

        out.compute();
        auto count = count_bytes_vals(out, kind);
        outputFile << "," << count.first << "," << count.second << std::endl;

        out.printComputeIR(std::cout);
    }

}
