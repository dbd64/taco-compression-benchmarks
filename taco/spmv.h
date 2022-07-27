#ifndef SPMV_H
#define SPMV_H

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

inline bool useRLEVector = false;

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

inline Func Mul_intersect("mul_intersect", MulOp(), intersectionAlgebra());
inline Func Mul_union("mul_union", MulOp(), unionAlgebra());
inline Func Mul_universe("mul_universe", MulOp(), universeAlgebra());

std::pair<Tensor<int>, Tensor<int>> load_sketches_grey(std::string path, int numImgs, std::string name, Kind kind, int64_t& numVals, int64_t& numBytes, bool transpose = true, int width=1111, int height=1111, int start=1);
std::pair<Tensor<int>, Tensor<int>> load_imgnet_grey(std::string path, std::string name, Kind kind, int64_t& numVals, int64_t& numBytes, bool transpose = true);

template <class T>
Tensor<T> makeRLEVector(std::string name, std::vector<T> v, int size){
    Tensor<T> t{name, {size}, {RLE}, 0};
    T curr = v[0];
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

template<class T>
using uniform_distribution = 
typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        void
    >::type
>::type;


template <class T>
Tensor<T> rand_vec(int index, int sz, int& numVals){
    std::default_random_engine gen(index);

    uniform_distribution<T> unif_vals_vec(1, 255);

    std::vector<T> v;
    int numValsVec = 1;
    for (int i=0; i<sz; i++){
        int label = unif_vals_vec(gen);
        numVals++;
        v.push_back(label);
    }
    Tensor<T> vec = useRLEVector ? makeRLEVector("vec_rand", v, v.size()) : makeDenseVector("vec_rand", v.size(), v);
    return vec;
}

template <class T>
Tensor<T> rand_matrix(int seed, int nrows, int ncols, Kind kind, int64_t& numVals, int64_t& numBytes){
    std::default_random_engine gen(seed);
    
    auto run_hist = getEnvVar("RUN_HIST");
    run_hist = run_hist == "" ? "/data/scratch/danielbd/artifact/out/spmv_temp/hist/rlep_hist_covtype.csv" : run_hist;
    std::cout << run_hist << std::endl;
    rapidcsv::Document run_hist_csv(run_hist, rapidcsv::LabelParams(0, -1));
    auto run_values  = run_hist_csv.GetColumn<int64_t>(0);
    auto run_weights = run_hist_csv.GetColumn<int64_t>(1);
    std::vector<double> run_weights_d;
    int64_t num_run_vals = 0;
    for (size_t i=0; i<run_values.size(); i++){
        num_run_vals += run_values[i]*run_weights[i];
    }


    auto lit_hist = getEnvVar("LIT_HIST");
    lit_hist = lit_hist == "" ? "/data/scratch/danielbd/artifact/out/spmv_temp/hist/rlep_lit_hist_covtype.csv" : lit_hist;
    std::cout << lit_hist << std::endl;
    rapidcsv::Document lit_hist_csv(lit_hist, rapidcsv::LabelParams(0, -1));
    auto lit_values  = lit_hist_csv.GetColumn<int64_t>(0);
    auto lit_weights = lit_hist_csv.GetColumn<int64_t>(1);
    std::vector<double> lit_weights_d;
    int64_t num_lit_vals = 0;
    for (size_t i=0; i<lit_values.size(); i++){
        num_lit_vals += lit_values[i]*lit_weights[i];
    }

    for (size_t i=0; i<run_weights.size(); i++){
        run_weights_d.push_back(run_weights[i]/((double) num_run_vals));
    }
    for (size_t i=0; i<lit_weights.size(); i++){
        lit_weights_d.push_back(lit_weights[i]/((double) num_lit_vals));
    }

    double temp0 = num_run_vals;
    double temp1 = num_lit_vals;

    std::discrete_distribution<int64_t> dist_choice({temp0, temp1});
    std::discrete_distribution<int64_t> run_dist(run_weights_d.begin(), run_weights_d.end());
    std::discrete_distribution<int64_t> lit_dist(lit_weights_d.begin(), lit_weights_d.end());
    uniform_distribution<T> unif_vals_vec(-1000, 1000);


    std::cout << "dist_choice: " << num_run_vals << ", " << num_lit_vals << std::endl;
    std::cout << "    " << dist_choice << std::endl;

    int run_br_cnt = 0;
    int lit_br_cnt = 0;
    std::vector<T> mtx;
    mtx.reserve(nrows*ncols);
    for (int row=0; row<nrows; row++){
        int col = 0;
        while (col < ncols){
            if (dist_choice(gen) == 0){
                run_br_cnt++;
                int64_t run_len = std::min(run_values[run_dist(gen)], int64_t(ncols-col));
                T val = unif_vals_vec(gen);
                for (int64_t i=0; i< run_len; i++){
                    mtx.push_back(val);
                }
                col += run_len;
            } else {
                lit_br_cnt++;
                int64_t lit_len = std::min(lit_values[lit_dist(gen)], int64_t(ncols-col));
                for (int64_t i=0; i< lit_len; i++){
                    T val = unif_vals_vec(gen);
                    while (mtx.size() > 0 && mtx.back() == val){
                        val = unif_vals_vec(gen);
                    }
                    mtx.push_back(val);
                }
                col += lit_len;
            }
        }
    }

    std::cout << "BR STATS : RUN " << run_br_cnt << ", LIT " << lit_br_cnt << std::endl;
    std::cout << mtx.size() << std::endl;

    auto mat = to_tensor_type<T>(mtx, nrows, ncols, 0, "mtx_", kind, numVals);
    numBytes = mat.second;
    return mat.first;
}

template <class T>
std::pair<Tensor<T>, Tensor<T>> load_csv(std::string filename, std::string name, Kind kind, int64_t& numVals, int64_t& numBytes, bool first, bool has_header = false, int offset = 1, bool transpose = true){
    rapidcsv::Document doc(filename, rapidcsv::LabelParams(has_header ? 0 : -1, -1));

    int col_lower = 0;
    int col_upper = doc.GetColumnCount();

    // Read values for vector
    std::cout << "Reading Vec" << std::endl;
    std::vector<T> v;
    if (first){
        col_lower += offset;
        v = doc.GetColumn<T>(0);
    } else {
        col_upper -= offset;
        v = doc.GetColumn<T>(doc.GetColumnCount()-1);
    }
    Tensor<T> vec = useRLEVector ? makeRLEVector("vec_" + name, v, v.size()) : makeDenseVector("vec_" + name, v.size(), v);

    // Read values for matrix
    if (transpose){
        int numRows = col_upper - col_lower;
        int numCols = doc.GetRowCount();
        std::vector<T> matrix;
        // for (int i=0; i< num)
        for (int i = col_lower; i < col_upper; i++){
            std::cout << "Col: " << i << std::endl;
            auto column = doc.GetColumn<T>(i);
            matrix.insert(matrix.end(), column.begin(), column.end());
        }
        auto mat = to_tensor_type<T>(matrix, numRows, numCols, 0, "mtx_" + name, kind, numVals);
        numBytes = mat.second;
        return {vec, mat.first};
    } else {
        int numCols = col_upper - col_lower;
        std::cout << "col_upper: " << col_upper << std::endl;
        std::cout << "col_lower: " << col_lower << std::endl;
        std::cout << "numCols: " << numCols << std::endl;
        std::cout << "doc.GetColumnCount() - col_upper: " << doc.GetColumnCount() - col_upper << std::endl;
        int numRows = doc.GetRowCount();
        std::vector<T> matrix;
        // for (int i=0; i< num)
        for (int i = 0; i < numRows; i++){
            auto column = doc.GetRow<T>(i);
            matrix.insert(matrix.end(), column.begin() + col_lower, column.end() - (doc.GetColumnCount() - col_upper));
        }
        auto mat = to_tensor_type<T>(matrix, numRows, numCols, 0, "mtx_" + name, kind, numVals);
        numBytes = mat.second;
        return {vec, mat.first};

    }
}

template <class T>
void bench_spmv(){
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
        f = Format{Dense, LZ77};
        func = Mul_universe;
    } else if (bench_kind == "RLEP"){
        kind = Kind::RLEP;
        f = Format{Dense, RLEP};
        func = Mul_universe;
    }


    auto data = getEnvVar("DATA_SOURCE");
    Tensor<T> vector, matrix;
    int64_t numVals = 0;
    int64_t numBytes = 0;
    bool useLanka = isLanka();
    bool transpose = true;
    auto lanka_root = "/data/scratch/danielbd/spmv_data/";
    auto laptop_root = "/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data/spmv/";
    if (data == "covtype"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("covtype.data");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "covtype", kind, numVals, numBytes, false);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "mnist"){
        transpose = false;
        auto csv = load_csv<T>((useLanka ? lanka_root : laptop_root) + std::string("mnist_train.csv"), "mnist", kind, numVals, numBytes, true, false, 1, transpose);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "sketches"){
        transpose = false;
        auto lanka_folder = "/data/scratch/danielbd/python_png_analysis/sketches/nodelta/";
        auto laptop_folder = "/Users/danieldonenfeld/Developer/png_analysis/sketches/nodelta/";
        auto csv = load_sketches_grey((useLanka ? lanka_folder : laptop_folder), 1000, "sketches", kind, numVals, numBytes);
        vector = csv.first;
        matrix = csv.second;
        if (kind == Kind::SPARSE) func = Mul_universe;
    } else if (data == "ilsvrc"){
        transpose = false;
        auto lanka_folder = lanka_root + std::string("cropped_grey/");
        auto laptop_folder = "/Users/danieldonenfeld/Developer/png_analysis/ILSVRC/Data/DET/cropped_grey/";
        auto csv = load_imgnet_grey((useLanka ? lanka_folder : laptop_folder), "ILSVRC", kind, numVals, numBytes);
        vector = csv.first;
        matrix = csv.second;
        if (kind == Kind::SPARSE) func = Mul_universe;
    } else if (data == "census"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("USCensus1990.data.csv");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "census", kind, numVals, numBytes, true, true);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "spgemm"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("sgemm_product.csv");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "spgemm", kind, numVals, numBytes, false, true, 4);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "poker"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("poker-hand-training-true.csv");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "poker", kind, numVals, numBytes, false, false, 0);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "kddcup"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("kddcup_processed.csv");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "kddcup", kind, numVals, numBytes, false, false, 0);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "hwd_plus"){
        transpose = false;
        auto lanka_folder = lanka_root + std::string("HWD_plus/");
        auto laptop_folder = "/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data/spmv/HWD_plus/";
        auto csv = load_sketches_grey((useLanka ? lanka_folder : laptop_folder), 8000 /*13579*/, "hwd_plus", kind, numVals, numBytes, transpose, 500, 500, 0);
        vector = csv.first;
        matrix = csv.second;
        if (kind == Kind::SPARSE) func = Mul_universe;
    } else if (data == "power"){
        auto file_path = (useLanka ? lanka_root : laptop_root) + std::string("household_power_consumption.csv");
        std::cout << "Reading from: " << file_path << std::endl;
        auto csv = load_csv<T>(file_path, "power", kind, numVals, numBytes, false, true, 0);
        vector = csv.first;
        matrix = csv.second;
    } else if (data == "random") {
        auto row_s = getEnvVar("NROWS");
        int nrows = row_s == "" ? 100 :  std::stoi(row_s);
        auto col_s = getEnvVar("NCOLS");
        int ncols = col_s == "" ? 100000 :  std::stoi(col_s);

        auto colwise = getEnvVar("COLWISE");
        if (colwise == "OFF") { transpose = false; }

        matrix = rand_matrix<T>(0, nrows, ncols, kind, numVals, numBytes);
    }

    std::string file_name = "/data/scratch/danielbd/artifact/tcb/taco/generated_code/";
    file_name = file_name + bench_kind + "_" + (transpose ? "col" : "row");

    if (bench_kind == "SPARSE" && (data == "sketches" || data == "ilsvrc")){
        file_name += "_spnz";
    } 
    if (data == "kddcup"|| data == "power"){
        file_name += "_float";
    } 

    file_name += ".c";

    std::string name = std::string(useRLEVector ? "RLE_" : "DENSE_") + "spmv_" + cache_str + "_" + bench_kind + "_" + data + ".csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeader(outputFile, repetitions);

    int temp = 0;
    vector = rand_vec<T>(0, matrix.getStorage().getDimensions()[transpose ? 0 : 1], temp);

    std::cout << matrix.getStorage().getDimensions() << std::endl;
    std::cout << vector.getStorage().getDimensions() << std::endl;

    // Tensor<T> copy_out("copy_out", matrix.getStorage().getDimensions(), {Dense,Dense});
    // copy_out(i,j) = copy(matrix(i,j));
    // copy_out.setAssembleWhileCompute(true);
    // copy_out.compile();
    // copy_out.compute();
    // write(getOutputPath() + "rawmtx/" + data + "_" + bench_kind + "_rawmtx.tns", copy_out);

    Tensor<T> out("out", {matrix.getStorage().getDimensions()[transpose ? 1 : 0]}, {Dense});
    IndexStmt stmt;
    if (transpose){
        stmt = (out(i) = func(matrix(j,i), vector(j)));
    } else {
        stmt = (out(i) = func(matrix(i,j), vector(j)));
    }

    // if (kind == Kind::RLE){
    //     stmt = stmt.concretize();
    //     IndexVar j0("j0");
    //     IndexVar j1("j1");
    //     IndexVar i0("i0");
    //     IndexVar i1("i1");
    //     stmt = stmt.split(j, j0, j1, 4);
    //     stmt = stmt.split(i, i0, i1, 2000);
    //     stmt = stmt.reorder({j0, i0, j1, i1}); // ORIG: j0, j1, i0,i1
    // }

    bool use_source = useSourceFiles();


    out.setAssembleWhileCompute(true);

    if (use_source){
        auto m = out.getModule();
        std::ifstream t(file_name);
        std::stringstream buffer;
        buffer << t.rdbuf();
        out.compileSource(buffer.str());
        // m->setSource(buffer.str());
        // m->compile();
    } else {
        std::cout << "Compiling " << name << std::endl;
        out.compile();
        out.printComputeIR(std::cout);
    }
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

    // if (use_source) {
    //     out.compileSource();
    // }
    out.compute();
    outputFile << "," << out.getStorage().getDimensions()[0]*4 << "," << out.getStorage().getDimensions()[0] << std::endl;

    write(getOutputPath() + "rawmtx/" + data + "_" + bench_kind + ".tns", out);
}


#endif