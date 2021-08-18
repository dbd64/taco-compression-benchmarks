#ifndef UTILS_H
#define UTILS_H

#include "taco.h"
#include "taco/tensor.h"
#include "lodepng.h"
#include "png_reader.h"
#include "bench.h"
#include <fstream>
#include <iterator>
#include <variant>

using namespace taco;

inline Index makeDenseIndex_2(int s0, int s1) {
  return Index(CSR, {ModeIndex({makeArray({s0})}),
                     ModeIndex({makeArray({s1})})});
}

template<typename T>
inline TensorBase makeDense_2(const std::string& name, const std::vector<int>& dims,
                   const std::vector<T>& vals) {
  Tensor<T> tensor(name, dims, Format{Dense, Dense});
  auto storage = tensor.getStorage();
  storage.setIndex(makeDenseIndex_2(dims[0], dims[1]));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

inline Index makeDenseIndex_3(int s0, int s1, int s2) {
 return Index(CSR, {ModeIndex({makeArray({s0})}),
                    ModeIndex({makeArray({s1})}),
                    ModeIndex({makeArray({s2})})});
}

template<typename T>
inline TensorBase makeDense_3(const std::string& name, const std::vector<int>& dims,
                  const std::vector<T>& vals) {
 Tensor<T> tensor(name, dims, Format{Dense, Dense, Dense});
 auto storage = tensor.getStorage();
 storage.setIndex(makeDenseIndex_3(dims[0], dims[1], dims[2]));
 storage.setValues(makeArray(vals));
 tensor.setStorage(storage);
 return std::move(tensor);
}

inline ir::Expr ternaryOp(const ir::Expr& c, const ir::Expr& a, const ir::Expr& b){
  // c ? a : b
  ir::Expr a_b = ir::BinOp::make(a,b, " : ");
  return ir::BinOp::make(c, a_b, "(", " ? ", ")");
}

std::pair<std::vector<uint8_t>, int> encode_lz77(const std::vector<uint8_t> in);

std::vector<uint8_t> raw_image_ma(std::string filename, int& w, int& h);

std::pair<Tensor<uint8_t>, size_t> to_tensor_rgb(const std::vector<uint8_t> image, int h, int w,
                                            int index, std::string prefix, Kind kind, int& numVals);
std::pair<Tensor<uint8_t>, size_t> to_tensor(const std::vector<uint8_t> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int& numVals);
                                             
std::pair<Tensor<uint8_t>, size_t> read_movie_frame(std::string img_folder, std::string prefix, int index, Kind kind, int& w, int& h, int& numVals);

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path, int width, int height); 
uint32_t saveTensor_RGB(std::vector<unsigned char> valsVec, std::string path, int width, int height);

void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, bool isroi, std::string bench_kind, int index, std::string prefix);
void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, std::string bench_kind, int index, std::string prefix, bool is_roi);

std::pair<int,int> count_bytes_vals(Tensor<uint8_t> t);

std::string to_string(Kind k);

void writeHeader(std::ostream& os, int repetitions);


#endif