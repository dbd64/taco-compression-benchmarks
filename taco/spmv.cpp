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

using namespace taco;

struct MulOp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Requires 2 arguments";
    return ir::Mul::make(v[0], v[1]);
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

Func Mul_intersect("mul_intersect", MulOp(), intersectionAlgebra());
Func Mul_union("mul_union", MulOp(), unionAlgebra());
Func Mul_universe("mul_universe", MulOp(), universeAlgebra());


std::pair<Tensor<int>, Tensor<int>> load_sketches_grey(std::string path, int numImgs, std::string name, Kind kind, int& numVals, int& numBytes){
    std::vector<int> v;
    int label = 1;
    for (int i=1; i<=numImgs; i++){
        if ((i-1) % 80 == 0 ){
            label++;
        }
        v.push_back(label);
    }
    Tensor<int> vec = makeDenseVector("vec_" + name, v.size(), v);

    // Load matrix
    std::vector<int> m;
    int width = 1111;
    int height = 1111;
    for (int i=1; i<=numImgs; i++){
        auto imgPath = path + std::to_string(i) +".png";
        auto img = raw_image_grey(imgPath, width, height);
        m.insert(m.end(), img.begin(), img.end());
    }

    auto mat = to_tensor_int(m, width*height, numImgs, 0, "mtx_" + name, kind, numVals, 255);
    numBytes = mat.second;
    return {vec, mat.first};
}

std::pair<Tensor<int>, Tensor<int>> load_imgnet_grey(std::string path, std::string name, Kind kind, int& numVals, int& numBytes){
    int numImgs = 5484;
    std::vector<int> v;
    int label = 1;
    for (int i=1; i<=numImgs; i++){
        if ((i-1) % 80 == 0 ){
            label++;
        }
        v.push_back(label);
    }
    Tensor<int> vec = makeDenseVector("vec_" + name, v.size(), v);

    // Load matrix
    std::vector<int> m;
    int width = 256;
    int height = 256;
    for (int i=1; i<=numImgs; i++){
        auto imgPath = path + std::to_string(i) +".png";
        auto img = raw_image_grey(imgPath, width, height);
        m.insert(m.end(), img.begin(), img.end());
    }

    auto mat = to_tensor_int(m, width*height, numImgs, 0, "mtx_" + name, kind, numVals, 255);
    numBytes = mat.second;
    return {vec, mat.first};
}

std::pair<Tensor<int>, Tensor<int>> load_csv(std::string filename, std::string name, Kind kind, int& numVals, int& numBytes, bool first){
    rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1));

    int col_lower = 0;
    int col_upper = doc.GetColumnCount();

    // Read values for vector
    std::vector<int> v;
    if (first){
        col_lower++;
        v = doc.GetColumn<int>(0);
    } else {
        col_upper--;
        v = doc.GetColumn<int>(doc.GetColumnCount()-1);
    }
    Tensor<int> vec = makeDenseVector("vec_" + name, v.size(), v);

    // Read values for matrix
    int numRows = col_upper - col_lower;
    int numCols = doc.GetRowCount();
    std::vector<int> matrix;
    // for (int i=0; i< num)
    for (int i = col_lower; i < col_upper; i++){
        auto column = doc.GetColumn<int>(i);
        matrix.insert(matrix.end(), column.begin(), column.end());
    }
    auto mat = to_tensor_int(matrix, numRows, numCols, 0, "mtx_" + name, kind, numVals);
    numBytes = mat.second;
    return {vec, mat.first};
}

void bench_spmv(){
    bool time = true;
    auto copy = getCopyFunc();
    taco::util::TimeResults timevalue{};
    const IndexVar i("i"), j("j"), c("c");

    int repetitions = 100;

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
        func = Mul_universe;
    } else if (bench_kind == "LZ77"){
        kind = Kind::LZ77;
        f = Format{LZ77};
        func = Mul_universe;
    }

    auto data = getEnvVar("DATA_SOURCE");
    Tensor<int> vector, matrix;
    int numVals = 0;
    int numBytes = 0;
    if (data == "covtype"){
        auto csv = load_csv("/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data/spmv/covtype.data", "covtype", kind, numVals, numBytes, false);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "mnist"){
        auto csv = load_csv("/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data/spmv/mnist_train.csv", "covtype", kind, numVals, numBytes, true);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "sketches"){
        auto csv = load_sketches_grey("/Users/danieldonenfeld/Developer/png_analysis/sketches/nodelta/", 1000, "sketches", kind, numVals, numBytes);
        vector = csv.first;
        matrix = csv.second;
        if (kind == Kind::SPARSE) func = Mul_universe;
    } else if (data == "ilsvrc"){
        auto csv = load_imgnet_grey("/Users/danieldonenfeld/Developer/png_analysis/ILSVRC/Data/DET/cropped_grey/", "ILSVRC", kind, numVals, numBytes);
        vector = csv.first;
        matrix = csv.second;
        if (kind == Kind::SPARSE) func = Mul_universe;
    }

    std::string name = "spmv_" + cache_str + "_" + bench_kind + "_" + data + ".csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeader(outputFile, repetitions);

    std::cout << vector.getStorage().getDimensions() << std::endl;
    std::cout << matrix.getStorage().getDimensions() << std::endl;

    Tensor<uint8_t> out("out", {matrix.getStorage().getDimensions()[0]}, {Dense});
    IndexStmt stmt = (out(i) = func(matrix(i,j), vector(j)));

    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = matrix.getStorage();
    taco_tensor_t* a2 = vector.getStorage();

    outputFile << 0 << "," << bench_kind << "," << numVals << "," << numBytes  << ",";
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