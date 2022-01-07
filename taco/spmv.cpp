#include "spmv.h"

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


Tensor<int> to_lz77(int index, int width, int height, int run_upper, int& numVals, int& numBytes, 
           std::uniform_int_distribution<int>& unif_vals, std::uniform_int_distribution<int>& unif_runs,
           std::default_random_engine& gen){
    std::vector<uint8_t> bytes;
    std::vector<int> pos;
    pos.push_back(0);
    for (int i = 0; i< height; i++){
        std::vector<TempValue<int>> m;
        int label = unif_vals(gen);
        int run_len = unif_runs(gen);
        int remaining = width;
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
        bytes.insert(bytes.end(), packed.begin(), packed.end());
        pos.push_back(bytes.size());
    }
    numBytes = bytes.size();
    auto t = makeLZ77<int>("mtx_rand" + std::to_string(index),
                          {height,width}, pos, bytes);
    return t;
}

std::pair<Tensor<int>, Tensor<int>> gen_rand(int index, int width, int height, int run_upper, Kind kind, int& numVals, int& numBytes){
    std::default_random_engine gen(index);
    std::uniform_int_distribution<int> unif_vals(0, 255);
    std::uniform_int_distribution<int> unif_runs(1, run_upper);

    {
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
    }

    // Load matrix
    Tensor<int> mat;
    if (kind == Kind::LZ77){
        mat = to_lz77(index, width, height, run_upper, numVals, numBytes, unif_vals, unif_runs, gen);
    } else {
        std::vector<int> m;
        int label = unif_vals(gen);
        int run_len = unif_runs(gen);
        for (int i=0; i<width*height; i++){
            if (run_len == 0 ){
                label = unif_vals(gen);
                run_len = unif_runs(gen);
            }
            m.push_back(label);
            run_len--;
        }

        auto matr = to_tensor_int(m, height, width, 0, "mtx_rand", kind, numVals, 0);
        mat = matr.first;
        numBytes = matr.second;
    }


    std::uniform_int_distribution<int> unif_vals_vec(1, 255);
    std::vector<int> v;
    int label = unif_vals_vec(gen);
    int run_len = unif_runs(gen);
    int numValsVec = 1;
    for (int i=0; i<width; i++){
        if (run_len == 0 ){
            label = unif_vals_vec(gen);
            run_len = unif_runs(gen);
            numValsVec++;
        }
        v.push_back(label);
        run_len--;
    }
    Tensor<int> vec = useRLEVector ? makeRLEVector("vec_rand", v, v.size()) : makeDenseVector("vec_rand", v.size(), v);

    numVals += numValsVec;
    return {vec, mat};
}

std::pair<Tensor<int>, Tensor<int>> load_sketches_grey(std::string path, int numImgs, std::string name, Kind kind, int& numVals, int& numBytes, bool transpose){
    std::vector<int> v;
    int label = 1;
    for (int i=1; i<=numImgs; i++){
        if ((i-1) % 80 == 0 ){
            label++;
        }
        v.push_back(label);
    }
    Tensor<int> vec = useRLEVector ? makeRLEVector("vec_" + name, v, v.size()) : makeDenseVector("vec_" + name, v.size(), v);

    // Load matrix
    std::vector<int> m;
    int width = 1111;
    int height = 1111;
    for (int i=1; i<=numImgs; i++){
        auto imgPath = path + std::to_string(i) +".png";
        auto img = raw_image_grey(imgPath, width, height);
        m.insert(m.end(), img.begin(), img.end());
    }

    auto mat = to_tensor_int(m, numImgs, width*height, 0, "mtx_" + name, kind, numVals, 255);
    numBytes = mat.second;
    return {vec, mat.first};
}

std::pair<Tensor<int>, Tensor<int>> load_imgnet_grey(std::string path, std::string name, Kind kind, int& numVals, int& numBytes, bool transpose){
    int numImgs = 5484;
    std::vector<int> v;
    int label = 1;
    for (int i=1; i<=numImgs; i++){
        if ((i-1) % 80 == 0 ){
            label++;
        }
        v.push_back(label);
    }
    Tensor<int> vec = useRLEVector ? makeRLEVector("vec_" + name, v, v.size()) : makeDenseVector("vec_" + name, v.size(), v);

    // Load matrix
    std::vector<int> m;
    int width = 256;
    int height = 256;
    for (int i=1; i<=numImgs; i++){
        auto imgPath = path + std::to_string(i) +".png";
        auto img = raw_image_grey(imgPath, width, height);
        m.insert(m.end(), img.begin(), img.end());
    }

    auto mat = to_tensor_int(m, numImgs, width*height, 0, "mtx_" + name, kind, numVals, 255);
    numBytes = mat.second;
    return {vec, mat.first};
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

void bench_spmv_rand(){
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
        func = Mul_universe;
    } else if (bench_kind == "RLE"){
        kind = Kind::RLE;
        f = Format{Dense,RLE};
        func = Mul_universe;
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


    std::string name = std::string(useRLEVector ? "RLE_" : "DENSE_") + "spmv_" + cache_str + "_" + bench_kind + "_" + index_str + "_" + run_upper_str + "_RAND.csv";
    name = getOutputPath() + name;
    std::ofstream outputFile(name);
    std::cout << "Starting " << name << std::endl;

    writeHeaderSpmvRand(outputFile, repetitions);


    // for (int index=start; index<=end; index++)
    {
        int numVals = 0;
        int numBytes = 0;
        auto ins = gen_rand(index, width, height, run_upper, kind, numVals, numBytes);
        Tensor<int> vector = ins.first;
        Tensor<int> matrix = ins.second;

        std::cout << vector.getDimensions() << std::endl;
        std::cout << matrix.getDimensions() << std::endl;

        Tensor<uint8_t> out("out", {matrix.getStorage().getDimensions()[0]}, {Dense});
        IndexStmt stmt = (out(i) = func(matrix(i,j), vector(j)));

        out.setAssembleWhileCompute(true);
        out.compile();
        Kernel k = getKernel(stmt, out);

        taco_tensor_t* a0 = out.getStorage();
        taco_tensor_t* a1 = matrix.getStorage();
        taco_tensor_t* a2 = vector.getStorage();

        outputFile << index << "," << bench_kind << "," << width << "," << height << "," << run_upper << "," << numVals << "," << numBytes  << ",";
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
