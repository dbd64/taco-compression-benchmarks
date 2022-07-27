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

template <class T>
std::pair<int,int> count_bytes_vals(Tensor<T> t, Kind kind){
  if (kind == Kind::DENSE){
    int num = t.getStorage().getValues().getSize();
    return {num*sizeof(T), num};
  } else if (kind == Kind::LZ77){
    int numVals = 0;
    int numBytes = t.getStorage().getValues().getSize();
    uint8_t* raw_bytes = (uint8_t*) t.getStorage().getValues().getData();
    std::vector<uint8_t> raw;
    raw.assign(raw_bytes, raw_bytes + numBytes);
    unpackLZ77_bytes(raw, numVals, false);
    return {numBytes, numVals};
  } else if (kind == Kind::SPARSE){
    int numVals = t.getStorage().getValues().getSize();
    return {4* numVals + sizeof(T)*numVals, numVals};
  } else if (kind == Kind::RLE){
    int numVals = t.getStorage().getValues().getSize();
    return {4* numVals + sizeof(T)*numVals, numVals};
  }
  return {-1,-1};
}

std::string to_string(Kind k);

void writeHeader(std::ostream& os, int repetitions);

int getNumRepetitions(int r);
int getIntEnvVar(std::string s, int d);
bool isLanka(bool d = true);
bool useSourceFiles(bool d = true);

Index makeRLEPIndex(const std::vector<int>& rowptr);
Index makeRLEPImgIndex(const std::vector<int>& rowptr);

template<typename T>
TensorBase makeRLEP(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
  // taco_uassert(dimensions.size() == 1);
  std::vector<ModeFormatPack> fs;
  for (int i = 0; i< dimensions.size()-1; i++) fs.push_back(Dense);
  fs.push_back(RLEP);
  Tensor<T> tensor(name, dimensions, Format(fs));
  auto storage = tensor.getStorage();
  if (dimensions.size()==2) {
    storage.setIndex(makeRLEPImgIndex(pos));
  } else {
    storage.setIndex(makeRLEPIndex(pos));
  }
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}



// using header_type = uint8_t;
// using header_type_s = int8_t;

using header_type = uint16_t;
using header_type_s = int16_t;

const int header_max = std::numeric_limits<header_type>::max() + 1;
const int header_mid = std::numeric_limits<header_type>::max()/2;

// template <class T>
// std::pair<std::vector<uint8_t>, int64_t> encode_rlep(const std::vector<T> in, std::vector<int64_t>& runs,std::vector<int64_t>& lits) {
//   int64_t in_idx = 0;
//   int count_idx = -1;
//   int64_t numRaw = 0;

//   std::vector<uint8_t> out;

//   while( in_idx < in.size() ) {
//     int64_t run_idx = in_idx+1;
//     header_type run_len = 0;
//     if (run_idx < in.size()){
//       while(run_idx < in.size() && in[run_idx] == in[in_idx]){
//         run_idx++;
//       } // a a b 
//       run_len = std::min((uint64_t)(run_idx - in_idx), (uint64_t) header_mid);
//       if (run_len > 1){
//         push_type<header_type>(out, header_max-run_len);
//         push_type<T>(out, in[in_idx]);
//         in_idx+=run_len;
//         numRaw++;
//         count_idx = -1;
//         runs[run_len]++;
//         run_len = 0;
//       } 
//     }

//     int lit_len = 0;
//     while(((in_idx + 1 < in.size() && in[in_idx + 1] != in[in_idx]) || in_idx == in.size()-1 || run_len--) && in_idx < in.size()){
//       if (count_idx != -1 && load_type<header_type>((uint8_t*)&out[0],count_idx) == header_mid+1) {
//         count_idx = -1;
//       }
//       if (count_idx == -1) {
//         count_idx = out.size();
//         push_type<header_type_s>(out, -1);
//       }

//       store_type<header_type>((uint8_t*)&out[0], count_idx, load_type<header_type>((uint8_t*)&out[0],count_idx)+1);
//       push_type<T>(out, in[in_idx++]);
//       numRaw++;
//       lit_len++;
//     }
//     if (lit_len > 0) { lits[lit_len]++; }
//   }

//   return {out,numRaw};
// }

template <class T>
std::pair<std::vector<uint8_t>, int64_t> encode_rlep(const std::vector<T> in, std::vector<int64_t>& runs,std::vector<int64_t>& lits) {
  int64_t in_idx = 0;
  int count_idx = -1;
  int64_t numRaw = 0;

  std::vector<uint8_t> out;

  while( in_idx < in.size() ) {
    int64_t run_idx = in_idx+1;
    header_type run_len = 0;
    if (run_idx < in.size()){
      while(run_idx < in.size() && in[run_idx] == in[in_idx]){
        run_idx++;
      } // a a b 
      run_len = std::min((uint64_t)(run_idx - in_idx), (uint64_t) 0x7FFF);
      if (run_len > 7){
        push_type<header_type>(out, run_len | 1 << 15);
        push_type<T>(out, in[in_idx]);
        in_idx+=run_len;
        numRaw++;
        if (count_idx!=-1) {lits[load_type<header_type>((uint8_t*)&out[0],count_idx)]++;}
        count_idx = -1;
        runs[run_len]++;
        run_len = 0;
      } 
    }

    int lit_len = 0;
    while(((in_idx + 1 < in.size() && in[in_idx + 1] != in[in_idx]) || in_idx == in.size()-1 || run_len--) && in_idx < in.size()){
      if (count_idx != -1 && load_type<header_type>((uint8_t*)&out[0],count_idx) == 0x7FFF) {
        count_idx = -1;
        lits[0x7FFF]++;
      }
      if (count_idx == -1) {
        count_idx = out.size();
        push_type<header_type_s>(out, 0);
      }

      store_type<header_type>((uint8_t*)&out[0], count_idx, load_type<header_type>((uint8_t*)&out[0],count_idx)+1);
      push_type<T>(out, in[in_idx++]);
      numRaw++;
      lit_len++;
    }
    // if (lit_len > 0) { lits[lit_len]++; }
  }

  return {out,numRaw};
}


template <class T>
std::pair<Tensor<T>, size_t> to_tensor_type(const std::vector<T> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int64_t& numVals, T sparseVal = 0){
  if (kind == Kind::DENSE){
    auto t = makeDense_2(prefix+"dense_" + std::to_string(index), {h,w}, image);
    numVals = h*w;
    return {t, h*w*sizeof(T)};
  } else if (kind == Kind::LZ77){
    std::vector<int> pos = {0};
    std::vector<uint8_t> values;
    int64_t bound = std::pow(2.0, 15);
    std::vector<int64_t> runs(bound, 0);
    std::vector<int64_t> lits(bound, 0);


    for (int i =0; i< h; i++){
      // std::cout << "Row " << i << std::endl;
      std::vector<T> row = {image.begin() + i*w, image.begin() + (i+1)*w}; 
      auto encoded = encode_lz77<T>(row, runs, lits);
      numVals += encoded.second;
      values.insert(values.end(), encoded.first.begin(), encoded.first.end());
      pos.push_back((int)values.size());
    }
    std::cout << "RAW VALUES (" << values.size() << "): ";
    for (size_t i=0; i < std::min((size_t)100,values.size()); i++){
      std::cout << (int)values[i] << " ";
    }
    std::cout << std::endl;

    std::string name = "lz77_run_" + getEnvVar("DATA_SOURCE") + ".csv";
    name = getOutputPath() + "hist/" + name;
    std::ofstream rle_hist(name);
    rle_hist << "run length,count" << std::endl;
    for (int i=0; i< runs.size(); i++){
      rle_hist << i << "," << runs[i] << std::endl;
    }

    name = "lz77_lit_" + getEnvVar("DATA_SOURCE") + ".csv";
    name = getOutputPath() + "hist/" + name;
    std::ofstream lit_hist(name);
    lit_hist << "run length,count" << std::endl;
    for (int i=0; i< lits.size(); i++){
      lit_hist << i << "," << lits[i] << std::endl;
    }

    auto t = makeLZ77<T>(prefix+"lz77_" + std::to_string(index),
                          {h,w}, pos, values);
    return {t, values.size()*sizeof(T) + pos.size() * 4};
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
    return {t, t.getStorage().getSizeInBytes()};
  } else if (kind == Kind::RLE) {
    Tensor<T> t{prefix+"rle_" + std::to_string(index), {h,w}, {Dense,RLE}, 0};
    std::vector<int> counts(w,0);
    int run_len = 0;
    for (int row=0; row<h; row++){
      if (run_len != 0) counts[run_len]++;
      T curr = image[row*w];
      t(row,0) = curr;
      run_len = 1;
      for (int col=0; col<w; col++){
        if (image[row*w + col] != curr){
          if (run_len != 0) counts[run_len]++;
          curr = image[row*w + col];
          t(row,col) = curr;
          run_len = 1;
        } else {
          run_len++;
        }
      }
    }

    std::string name = "rle_hist_" + getEnvVar("DATA_SOURCE") + ".csv";
    name = getOutputPath() + name;
    std::ofstream rle_hist(name);
    rle_hist << "run length,count" << std::endl;
    for (int i=0; i< counts.size(); i++){
      rle_hist << i << "," << counts[i] << std::endl;
    }

    int size = std::min((int)counts.size(), 1200);
    std::cout << "[";
    for (int i=0; i< size-1; i++){
      std::cout << counts[i] << ", ";
    }
    if (size > 0) {std::cout << counts[size-1];}
    std::cout << "]\n";
    t.pack();
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getSizeInBytes()};
  } else if (kind == Kind::RLEP){
    std::cout << "RLEP CONSTS: max " << header_max << ", mid " << header_mid << std::endl;

    std::vector<int> pos = {0};
    std::vector<uint8_t> values;
    std::vector<int64_t> runs(header_mid+1,0);
    std::vector<int64_t> lits(w,0);

    for (int i =0; i< h; i++){
      // std::cout << "Row " << i << std::endl;
      std::vector<T> row = {image.begin() + i*w, image.begin() + (i+1)*w}; 
      auto encoded = encode_rlep<T>(row, runs, lits);
      numVals += encoded.second;
      values.insert(values.end(), encoded.first.begin(), encoded.first.end());
      pos.push_back((int)values.size());
    }
    std::cout << "RAW VALUES (" << values.size() << "): ";
    for (size_t i=0; i < std::min((size_t)100,values.size()); i++){
      std::cout << (int)values[i] << " ";
    }
    std::cout << std::endl;

    std::string name = "rlep_hist_" + getEnvVar("DATA_SOURCE") + ".csv";
    name = getOutputPath() + "hist/" + name;
    std::ofstream rle_hist(name);
    rle_hist << "run length,count" << std::endl;
    for (int i=0; i< runs.size(); i++){
      rle_hist << i << "," << runs[i] << std::endl;
    }

    name = "rlep_lit_hist_" + getEnvVar("DATA_SOURCE") + ".csv";
    name = getOutputPath() + "hist/" + name;
    std::ofstream lit_hist(name);
    lit_hist << "run length,count" << std::endl;
    for (int i=0; i< lits.size(); i++){
      lit_hist << i << "," << lits[i] << std::endl;
    }


    auto t = makeRLEP<T>(prefix+"rlep_" + std::to_string(index),
                          {h,w}, pos, values);
    return {t, t.getStorage().getSizeInBytes()};
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
    return {t, t.getStorage().getSizeInBytes()};
  }
  Tensor<T> t{"error", {w*h}, {Dense}};
  return {t,0};
}

#endif