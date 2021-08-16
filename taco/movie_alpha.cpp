#include "bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"
#include "taco/util/timers.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"
#include "png_reader.h"

using namespace taco;

Index makeDenseIndex_3(int s0, int s1, int s2) {
 return Index(CSR, {ModeIndex({makeArray({s0})}),
                    ModeIndex({makeArray({s1})}),
                    ModeIndex({makeArray({s2})})});
}

template<typename T>
TensorBase makeDense_3(const std::string& name, const std::vector<int>& dims,
                  const std::vector<T>& vals) {
 Tensor<T> tensor(name, dims, Format{Dense, Dense});
 auto storage = tensor.getStorage();
 storage.setIndex(makeDenseIndex_3(dims[0], dims[1], dims[2]));
 storage.setValues(makeArray(vals));
 tensor.setStorage(storage);
 return std::move(tensor);
}

std::vector<uint8_t> raw_image_ma(std::string filename, int& w, int& h){
   std::vector<unsigned char> png;
   std::vector<unsigned char> image; //the raw pixels
   std::vector<unsigned char> compressed;
   std::vector<int> pos;
   unsigned width = 0, height = 0;

   unsigned error = lodepng::load_file(png, filename);
   if(!error) error = decode(image, compressed, pos, width, height, png, LCT_RGB);

   if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

   w = (int)width;
   h = (int)height;
   return std::move(image);
}

std::vector<uint8_t> encode_lz77(const std::vector<uint8_t> in);

std::pair<Tensor<uint8_t>, size_t> to_tensor_rgb(const std::vector<uint8_t> image, int h, int w,
                                            int index, std::string prefix, Kind kind){
 if (kind == Kind::DENSE){
   auto t = makeDense_3(prefix+"dense_" + std::to_string(index), {h,w,3}, image);
   return {t, h*w};
 } else if (kind == Kind::LZ77){
   auto packed = encode_lz77(image);
   auto t = makeLZ77<uint8_t>(prefix+"lz77_" + std::to_string(index),
                         {h*w*3},
                         {0, (int)packed.size()}, packed);
   return {t, packed.size()};
 } else if (kind == Kind::SPARSE){
   Tensor<uint8_t> t{prefix+"sparse_" + std::to_string(index), {h,w,3}, {Dense,Sparse,Dense}, 0};
    for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
        if (image[row*w*3 + col*3 + 0] != 0 ||
            image[row*w*3 + col*3 + 1] != 0 ||
            image[row*w*3 + col*3 + 2] != 0){
            for (int color=0; color<3; color++){
            t(row,col,color) = image[row*w*3 + col*3 + color];
            }
        }
        }
    }
    t.pack();
   return {t, t.getStorage().getValues().getSize()};
 } else if (kind == Kind::RLE){
   Tensor<uint8_t> t{prefix+"rle_" + std::to_string(index), {h,w,3}, {Dense,RLE_size(3),Dense}, 0};
    uint8_t curr[3] = {image[0], image[1], image[2]};
    t(0,0,0) = curr[0];
    t(0,0,1) = curr[1];
    t(0,0,2) = curr[2];
    for (int row=0; row<h; row++){
    for (int col=0; col<w; col++){
        if (image[row*w*3 + col*3 + 0] != curr[0] ||
            image[row*w*3 + col*3 + 1] != curr[1] ||
            image[row*w*3 + col*3 + 2] != curr[2]){
        curr[0] = image[row*w*3 + col*3 + 0];
        curr[1] = image[row*w*3 + col*3 + 1];
        curr[2] = image[row*w*3 + col*3 + 2];
        for (int color=0; color<3; color++){
            t(row,col,color) = image[row*w*3 + col*3 + color];
        }
        }
    }
    }
    t.pack();
   return {t, t.getStorage().getValues().getSize()};
 }
}

std::pair<Tensor<uint8_t>, size_t> read_movie_frame(std::string img_folder, int index, Kind kind, int& w, int& h) {
  std::ostringstream stringStream;
  stringStream << img_folder;
  if (index < 10){
      stringStream << "00";
  } else if (index < 100){
      stringStream << "0";
  }
  stringStream << index << ".png";
  std::string filename = stringStream.str();

  auto image = raw_image_ma(filename, w, h);

 return to_tensor_rgb(image,h,w,index,"f" + std::to_string(index) + "_", kind);
}

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path, int width, int height){
 std::vector<unsigned char> png_mine;
 auto error = lodepng::encode(png_mine, valsVec, width, height, LCT_RGB);
 if(!error) lodepng::save_file(png_mine, path);
 if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
 return error;
}

void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, std::string bench_kind, int index, std::string prefix){
 const IndexVar i("i"), j("j"), c("c");
 auto copy = getCopyFunc();
 auto dims = roi_t.getDimensions();

 Tensor<uint8_t> v("v", dims, dims.size() == 1? Format{Dense} : Format{Dense,Dense});
 if (kind == Kind::LZ77){
   v(i) = copy(roi_t(i));
 } else {
   v(i,j,c) = copy(roi_t(i,j,c));
 }
 v.setAssembleWhileCompute(true);
 v.compile();
 v.compute();

 uint8_t* start = (uint8_t*) v.getStorage().getValues().getData();
 std::vector<uint8_t> validation(start, start+w*h);
 saveTensor(validation, getValidationOutputPath() + prefix + "_" + bench_kind+ "_" + std::to_string(index) + ".png",  w, h);
}

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


void movie_alpha_bench(){
 bool time = true;
 auto copy = getCopyFunc();
 taco::util::TimeResults timevalue{};
 const IndexVar i("i"), j("j"), c("c");

 int repetitions = 100;

 std::cout << "index,kind,total_bytes,mean,stddev,median" << std::endl;

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


 for (int index=start; index<=end; index++){
   int w = 0;
   int h = 0;
   auto f1 = read_movie_frame(folder1, index, kind, w, h);
   auto f2 = read_movie_frame(folder1, index, kind, w, h);
   auto dims = f1.first.getDimensions();

   auto f1t = f1.first;
   auto f2t = f2.first;

   Tensor<uint8_t> out("out", dims, f);
   IndexStmt stmt;
   if (kind == Kind::LZ77){
     stmt = (out(i) = 0.7*f1t(i) + 0.3*f2t(i));
   } else {
     stmt = (out(i,j,c) = 0.7*f1t(i,j,c) + 0.3*f2t(i,j,c));
   }
   out.setAssembleWhileCompute(true);
   out.compile();
   Kernel k = getKernel(stmt, out);

   taco_tensor_t* a0 = out.getStorage();
   taco_tensor_t* a1 = f1t.getStorage();
   taco_tensor_t* a2 = f2t.getStorage();

   std::cout << index << "," << bench_kind << "," << f1.second + f2.second  << ",";
   TOOL_BENCHMARK_REPEAT({
       k.compute(a0,a1,a2);
   }, "Compute", repetitions);

   out.compute();

   // out.printComputeIR(std::cout);

   saveValidation(f1t, kind, w, h, bench_kind, index, "f1");
   saveValidation(f2t, kind, w, h, bench_kind, index, "f2");
   saveValidation(out, kind, w, h, bench_kind, index, "out");
 }
}