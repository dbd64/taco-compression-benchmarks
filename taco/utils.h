#ifndef UTILS_H
#define UTILS_H

#include "taco.h"
#include "taco/tensor.h"
#include "lodepng.h"
#include "png_reader.h"
#include "bench.h"
#include "lz77_compress.h"
#include <fstream>
#include <iterator>
#include <variant>

using namespace taco;

inline Index makeDenseVectorIndex(int s0) {
  return Index({Dense}, {ModeIndex({makeArray({s0})})});
}

template<typename T>
inline TensorBase makeDenseVector(const std::string& name, const int& dim,
                   const std::vector<T>& vals) {
  Tensor<T> tensor(name, {dim}, Format{Dense});
  auto storage = tensor.getStorage();
  storage.setIndex(makeDenseVectorIndex(dim));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}


inline Index makeDenseIndex_2(int s0, int s1) {
  return Index({Dense, Dense}, {ModeIndex({makeArray({s0})}),
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
 return Index({Dense, Dense}, {ModeIndex({makeArray({s0})}),
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

// std::pair<std::vector<uint8_t>, int> encode_lz77(const std::vector<uint8_t> in);

std::vector<uint8_t> raw_image_ma(std::string filename, int& w, int& h);
std::vector<uint8_t> raw_image_grey(std::string filename, int& w, int& h);
std::vector<uint8_t> raw_image_subtitle(std::string filename, int& w, int& h);

std::pair<Tensor<uint8_t>, size_t> to_tensor_rgb(const std::vector<uint8_t> image, int h, int w,
                                            int index, std::string prefix, Kind kind, int& numVals);
std::pair<Tensor<uint8_t>, size_t> to_tensor(const std::vector<uint8_t> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int& numVals, int sparseVal = 0);
std::pair<Tensor<int>, size_t> to_tensor_int(const std::vector<int> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int& numVals, int sparseVal = 0);

std::pair<Tensor<uint8_t>, size_t> to_vector_rgb(const std::vector<uint8_t> image, int h, int w,
                                            int index, std::string prefix, Kind kind, int& numVals);

std::pair<Tensor<uint8_t>, size_t> read_movie_frame(std::string img_folder, std::string prefix, int index, Kind kind, int& w, int& h, int& numVals);
std::pair<Tensor<uint8_t>, Tensor<uint8_t>> read_subtitle_mask(Kind kind, int width, int height, int& maskBytes, int& maskVals, int& imgBytes, int& imgVals);

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path, int width, int height); 
uint32_t saveTensor_RGB(std::vector<unsigned char> valsVec, std::string path, int width, int height);

void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, bool isroi, std::string bench_kind, int index, std::string prefix);
void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, std::string bench_kind, int index, std::string prefix, bool is_roi);

std::pair<int,int> count_bytes_vals(Tensor<uint8_t> t, Kind kind);

std::string to_string(Kind k);

void writeHeader(std::ostream& os, int repetitions);

int getNumRepetitions(int r);
int getIntEnvVar(std::string s, int d);
bool isLanka(bool d = true);

template <class T>
std::pair<Tensor<T>, size_t> to_tensor_type(const std::vector<T> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int& numVals, T sparseVal = 0){
  if (kind == Kind::DENSE){
    auto t = makeDense_2(prefix+"dense_" + std::to_string(index), {h,w}, image);
    numVals = h*w;
    return {t, h*w*sizeof(T)};
  } else if (kind == Kind::LZ77){
    std::vector<int> pos;
    std::vector<uint8_t> values;
    for (int i =0; i< h; i++){
      std::vector<T> row = {image.begin() + i*w, image.begin() + (i+1)*w}; 
      auto encoded = encode_lz77<T>(row);
      numVals += encoded.second;
      values.insert(values.end(), encoded.first.begin(), encoded.first.end());
      pos.push_back((int)values.size());
    }
    auto t = makeLZ77<T>(prefix+"lz77_" + std::to_string(index),
                          {h,w}, pos, values);
    return {t, values.size() + pos.size() * sizeof(T)};
  } else if (kind == Kind::SPARSE){
    Tensor<T> t{prefix+"sparse_" + std::to_string(index), {h,w}, {Dense,Sparse}, sparseVal};
    for (int row=0; row<h; row++){
      for (int col=0; col<w; col++){
        if (image[row*w + col] != sparseVal){
            t(row,col) = image[row*w + col];
        }
      }
    }
    t.pack();
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getValues().getSize()*sizeof(T)};
  } else if (kind == Kind::RLE){
    Tensor<T> t{prefix+"rle_" + std::to_string(index), {h,w}, {Dense,RLE}, 0};
    T curr = image[0];
    t(0,0) = curr;
    for (int row=0; row<h; row++){
      for (int col=0; col<w; col++){
        if (image[row*w + col] != curr){
          curr = image[row*w + col];
          t(row,col) = curr;
        }
      }
    }
    t.pack();
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getValues().getSize()*sizeof(T)};
  }
  Tensor<T> t{"error", {w*h}, {Dense}};
  return {t,0};
}

#endif