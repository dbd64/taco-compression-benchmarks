#include "lodepng.h"

#include "../test/test.h"
#include "taco/tensor.h"
#include "taco/util/timers.h"
#include "src/lower/iteration_graph.h"
#include "taco.h"

#include "bench.h"
#include "benchmark/benchmark.h"


#include <benchmark/benchmark.h>

#include <iostream>
#include <random>
#include <variant>
#include <climits>
#include <limits>
#include <fstream>

extern "C" {
#include "lz_sum_kernel.h"
}

#include "png_reader.h"

using namespace taco;


/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned ucvector_resize(ucvector* p, size_t size) {
  if(size > p->allocsize) {
    size_t newsize = size + (p->allocsize >> 1u);
    void* data = realloc(p->data, newsize);
    if(data) {
      p->allocsize = newsize;
      p->data = (unsigned char*)data;
    }
    else return 0; /*error: not enough memory*/
  }
  p->size = size;
  return 1; /*success*/
}

static ucvector ucvector_init(unsigned char* buffer, size_t size) {
  ucvector v;
  v.data = buffer;
  v.allocsize = v.size = size;
  return v;
}

unsigned lodepng_zlib_decompressv(ucvector* out,
                                         const unsigned char* in, size_t insize,
                                         const LodePNGDecompressSettings* settings);

unsigned lodepng_inflatev(ucvector* out,
                          const unsigned char* in, size_t insize,
                          const LodePNGDecompressSettings* settings);

unsigned custom_zlib_f(unsigned char** out, size_t* outsize,
                          const unsigned char* in, size_t insize,
                          const LodePNGDecompressSettings* settings){
  ucvector v = ucvector_init(*out, *outsize);
  unsigned error = lodepng_zlib_decompressv(&v, in, insize, settings);
  *out = v.data;
  *outsize = v.size;
  return error;
}

unsigned custom_inflate_f(unsigned char ** out, size_t* outsize,
                                 const unsigned char* in, size_t insize,
                                 const LodePNGDecompressSettings* settings);

void* lodepng_malloc(size_t size);

unsigned lodepng_decode_memory_(unsigned char** out, unsigned char** compressed,
                                unsigned* compressed_size,
                                std::vector<int>& pos,
                                unsigned* w, unsigned* h, const unsigned char* in,
                               size_t insize, LodePNGColorType colortype, unsigned bitdepth) {
  unsigned error;
  LodePNGState state;
  lodepng_state_init(&state);
  state.info_raw.colortype = colortype;
  state.info_raw.bitdepth = bitdepth;
  state.decoder.zlibsettings.custom_zlib = custom_zlib_f;
  state.decoder.zlibsettings.custom_inflate = custom_inflate_f;
  state.decoder.zlibsettings.ignore_adler32 = true;
  state.decoder.zlibsettings.custom_context = new custom_context_s;
//          lodepng_malloc(sizeof(custom_context_s));

//  ((custom_context_s*)state.decoder.zlibsettings.custom_context)->pos = std::vector<int>();
  ((custom_context_s*)state.decoder.zlibsettings.custom_context)->pos.push_back(0);

  error = lodepng_decode(out, w, h, &state, in, insize);

  *compressed = ((custom_context_s*)state.decoder.zlibsettings.custom_context)->compressed.data;
  *compressed_size = ((custom_context_s*)state.decoder.zlibsettings.custom_context)->compressed.size;
  pos = ((custom_context_s*)state.decoder.zlibsettings.custom_context)->pos;

  lodepng_state_cleanup(&state);
  return error;
}

unsigned decode(std::vector<unsigned char>& out, std::vector<unsigned char>& compressed, std::vector<int>& pos, unsigned& w, unsigned& h, const unsigned char* in,
                size_t insize, LodePNGColorType colortype, unsigned bitdepth) {
  unsigned char* buffer = 0;
  unsigned char* c = 0;
  unsigned cs = 0;
  unsigned error = lodepng_decode_memory_(&buffer, &c, &cs, pos, &w, &h, in, insize, colortype, bitdepth);
  if(buffer && !error) {
    lodepng::State state;
    state.info_raw.colortype = colortype;
    state.info_raw.bitdepth = bitdepth;
    size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
  }
  if(c && !error){
    compressed.insert(compressed.end(), &c[0], &c[cs]);
  }
  free(buffer);
  return error;
}


unsigned decode(std::vector<unsigned char>& out, std::vector<unsigned char>& c_out, std::vector<int>& pos, unsigned& w, unsigned& h,
                const std::vector<unsigned char>& in, LodePNGColorType colortype, unsigned bitdepth) {
  return decode(out, c_out, pos, w, h, in.empty() ? 0 : &in[0], (unsigned)in.size(), colortype, bitdepth);
}

std::vector<uint8_t> unpackLZ77_bytes(std::vector<uint8_t> bytes, int& numVals, bool unpack){
  // 0 XXXXXXX XXXXXXXX     -> read X number of bytes
  // 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

  std::vector<uint8_t> out;
  size_t i = 0;
  while (i<bytes.size()){
    if ((bytes[i+1] >> 7 & 1) == 0){
      uint16_t numBytes = *((uint16_t *) &bytes[i]);
      i+=2;
      for(int j=0; j<numBytes; j++){
        if (unpack) out.push_back(bytes[i+j]);
        numVals++;
      }
      i+=numBytes;
    } else  {
      uint16_t dist = *((uint16_t *) &bytes[i+2]);
      uint16_t run = *((uint16_t *) &bytes[i]) & (uint16_t)0x7FFF;
      if (unpack){
        if (dist <= run){
          for (size_t j = i-dist; j<i; j++){
            out.push_back(bytes[j]);
          }
          size_t start = out.size()-dist;
          for (size_t j = 0; j<(run-dist); j++){
            out.push_back(out[start + j]);
          }
        } else {
          for (size_t j = i-dist; j < i-dist+run; j++){
            out.push_back(bytes[j]);
          }
        }
      }
      i+=4;
    }
  }
  return out;
}

std::vector<TempValue<uint8_t>> unpackLZ77_vector(std::vector<uint8_t> bytes){
  // 0 XXXXXXX XXXXXXXX     -> read X number of bytes
  // 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

  std::vector<TempValue<uint8_t>> out;
  size_t i = 0;
  while (i<bytes.size()){
    if ((bytes[i+1] >> 7 & 1) == 0){
      uint16_t numBytes = *((uint16_t *) &bytes[i]);
      i+=2;
      for(int j=0; j<numBytes; j++){
        out.emplace_back(bytes[i+j]);
      }
      i+=numBytes;
    } else  {
      uint16_t dist = *((uint16_t *) &bytes[i+2]);
      uint16_t run = *((uint16_t *) &bytes[i]) & (uint16_t)0x7FFF;
      out.push_back(Repeat{dist,run});
      i+=4;
    }
  }
  return out;
}



void print_bytes(const std::vector<unsigned char>& image, unsigned width, unsigned height){
  for(int i = 0; (size_t)i < image.size(); i++){
    if (i % width == 0){
      std::cout << std::endl;
    }
    std::cout << (unsigned)image[i] << ", ";
  }
  std::cout << std::endl;
}

Func getCopyFunc(){
  auto copyFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], 0);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      return Union(l, Complement(l));
  };
  Func copy("copy_", copyFunc, algFunc);
  return copy;
}

Func getPlusFunc(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

Func getPlusRleFunc(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(l, r);
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

std::vector<TempValue<uint8_t>> compress(std::vector<uint8_t> vals){
  // Do RLE compression on vals
  std::vector<TempValue<uint8_t>> out;
  out.emplace_back(vals[0]);
  uint16_t r = 0;
  uint8_t c = vals[0];
  for (size_t i=1; i<vals.size(); i++){
    if (vals[i] == c && r+1<32768){
      r++;
    } else {
      if (r){
        out.push_back(Repeat{1,r});
        r = 0;
      }
      c = vals[i];
      out.emplace_back(c);
    }
  }
  if (r){
    out.push_back(Repeat{1,r});
    r = 0;
  }
  return out;
}


template <typename T>
std::pair<std::vector<uint8_t>, int> packLZ77(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes = packLZ77_bytes(vals);
  return {bytes,bytes.size()};
//  int size = bytes.size();
//  while(bytes.size() % sizeof(T) != 0){
//    bytes.push_back(0);
//  }
//  T* bytes_data = (T*) bytes.data();
//  std::vector<T> values(bytes_data, bytes_data + (bytes.size() / sizeof(T)));
//
//  return {values, size};
}

Index makeLZ77Index(const std::vector<int>& rowptr) {
  return Index({LZ77},
               {ModeIndex({makeArray(rowptr)})});
}

Index makeLZ77ImgIndex(const std::vector<int>& rowptr) {
  return Index({Dense, LZ77},
               {ModeIndex({makeArray({(int)rowptr.size()})}),
                ModeIndex({makeArray(rowptr)})});
}

template<typename T>
TensorBase makeLZ77Img(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
  taco_uassert(dimensions.size() == 2);
  Tensor<T> tensor(name, dimensions, {Dense, LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77ImgIndex(pos));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

string toStr(int i){
  std::ostringstream stringStream;
  stringStream << i;
  return stringStream.str();
}

std::pair<Tensor<uint8_t>, size_t> read_png(int i, Kind kind) {
  auto img_folder = getEnvVar("IMAGE_FOLDER");
  if (img_folder == "") {
    img_folder = "/Users/danieldonenfeld/Developer/png_analysis/sketches/nodelta/";
  }

  std::vector<unsigned char> png;
  std::vector<unsigned char> image; //the raw pixels
  std::vector<unsigned char> compressed;
  std::vector<int> pos;
  unsigned width = 0, height = 0;

  std::ostringstream stringStream;
  stringStream << img_folder;
  stringStream << i << ".png";
  std::string filename = stringStream.str();

  //load and decode
  unsigned error = lodepng::load_file(png, filename);
  if(!error) error = decode(image, compressed, pos, width, height, png);

  //if there's an error, display it
  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  if(packLZ77_bytes(unpackLZ77_vector(compressed)) != compressed){
    std::cout << "packLZ77_bytes(unpackLZ77_vector()) not working" << std::endl;
  }

  int numVals = 0;
  auto temp = unpackLZ77_bytes(compressed, numVals);
  //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...

  auto packed_rle = packLZ77_bytes(compress(image));
  if(unpackLZ77_bytes(packed_rle, numVals) != image){
    std::cout << "unpackLZ77_bytes(packLZ77_bytes(compress(image))) not working" << std::endl;
  } else {
//    std::cout << "packed_rle bytes: " << packed_rle.size() << std::endl;
//    compressed = packed_rle;
  }


  if(!(image == temp)) {
    std::vector<int> badPx;
    for (int p = 0; p<image.size(); p++){
      if (image[p] != temp[p]){
//        std::cout << (unsigned)image[p] << ", " << (unsigned)temp[p] << std::endl;
        badPx.push_back(p);
      }
    }
    std::cout << "Raw != unpacked (" << i << "): " << badPx << std::endl;



    std::vector<unsigned char> png_mine;

    // error = lodepng::encode(png_mine, temp, width, height, LCT_GREY);
    // if(!error) lodepng::save_file(png_mine, "/Users/danieldonenfeld/Developer/taco/apps/png_reader/test_out.png");

    // //if there's an error, display it
    // if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  }

//  print_bytes(image, width, height);
//  print_bytes(temp, width, height);

//  std::cout <<  png.size()  << "," << image.size() << "," << compressed.size() <<  std::endl;

  if (kind == Kind::DENSE){
    int w = (int)width;
    int h = (int)height;
    Tensor<uint8_t> t{"T" + toStr(i), {w*h}, {Dense}};
    for (int j=0; j<h; j++){
      for (int k=0; k<w; k++){
//        t(j,k) = image[j*h + k];
        t(j*h + k) = image[j*h + k];
      }
    }
    return {t,w*h};
  } else if (kind == Kind::LZ77){
//    return {makeLZ77<uint8_t>("T" + toStr(i),
//                             {(int)height*(int)width},
//                             {0, pos.back()}, compressed),
//            compressed.size()};
    return {makeLZ77<uint8_t>("T" + toStr(i),
                              {(int)height*(int)width},
                              {0, (int)packed_rle.size()}, packed_rle),
            packed_rle.size()};

  } else if (kind == Kind::SPARSE){
    int w = (int)width;
    int h = (int)height;
    Tensor<uint8_t> t{"T" + toStr(i), {w*h}, {Sparse}, 255};
    for (int j=0; j<h; j++){
      for (int k=0; k<w; k++){
        if(image[j*h + k] != 255){
          t(j*h + k) = image[j*h + k];
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  } else if (kind == Kind::RLE){
    int w = (int)width;
    int h = (int)height;
    Tensor<uint8_t> t{"T" + toStr(i), {w*h}, {RLE}, 255};
    uint8_t curr = image[0];
    t(0) = curr;
//    for (int j=0; j<h; j++){
//      for (int k=0; k<w; k++){
//          t(j*h + k) = image[j*h+k];
//      }
//    }

    for (int p=1; p<w*h; p++){
//      t(p) = image[p];
      if (image[p] != curr){
        curr = image[p];
        t(p) = curr;
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  }

}

std::pair<Tensor<uint8_t>, size_t> read_rgb_png(int i, Kind kind) {
  std::vector<unsigned char> png;
  std::vector<unsigned char> image; //the raw pixels
  std::vector<unsigned char> compressed;
  std::vector<int> pos;
  unsigned width = 0, height = 0;

  std::ostringstream stringStream;
  stringStream << "/Users/danieldonenfeld/Developer/png_analysis/bars_and_stripes_png/";
  if (i < 10){
    stringStream << "00";
  } else if (i < 100){
    stringStream << "0";
  }
  stringStream << i << ".png";
  std::string filename = stringStream.str();

  //load and decode
  unsigned error = lodepng::load_file(png, filename);
  if(!error) error = decode(image, compressed, pos, width, height, png, LCT_RGB);

  //if there's an error, display it
  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

//  std::cout << "image sizimage sizee : " << image.size() << endl;
//  std::cout << "compressed size : " << compressed.size() << endl;

  std::vector<unsigned char> png_mine;

  error = lodepng::encode(png_mine, image, width, height, LCT_RGB);
  if(!error) lodepng::save_file(png_mine, "/Users/danieldonenfeld/Developer/taco/apps/png_reader/test_out_rgb.png");

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;


//  if(packLZ77_bytes(unpackLZ77_vector(compressed)) != compressed){
//    std::cout << "packLZ77_bytes(unpackLZ77_vector()) not working" << std::endl;
//  }
//
//  auto temp = unpackLZ77_bytes(compressed);
//  //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
//
//  auto packed_rle = packLZ77_bytes(compress(image));
//  if(unpackLZ77_bytes(packed_rle) != image){
//    std::cout << "unpackLZ77_bytes(packLZ77_bytes(compress(image))) not working" << std::endl;
//  } else {
////    std::cout << "packed_rle bytes: " << packed_rle.size() << std::endl;
////    compressed = packed_rle;
//  }


//  if(!(image == temp)) {
//    std::vector<int> badPx;
//    for (int p = 0; p<image.size(); p++){
//      if (image[p] != temp[p]){
////        std::cout << (unsigned)image[p] << ", " << (unsigned)temp[p] << std::endl;
//        badPx.push_back(p);
//      }
//    }
//    std::cout << "Raw != unpacked (" << i << "): " << badPx << std::endl;
//
//
//
//    std::vector<unsigned char> png_mine;
//
//    error = lodepng::encode(png_mine, temp, width, height, LCT_GREY);
//    if(!error) lodepng::save_file(png_mine, "/Users/danieldonenfeld/Developer/taco/apps/png_reader/test_out.png");
//
//    //if there's an error, display it
//    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
//  }

//  print_bytes(image, width, height);
//  print_bytes(temp, width, height);

//  std::cout <<  png.size()  << "," << image.size() << "," << compressed.size() <<  std::endl;

  int w = (int)width;
  int h = (int)height;
  if (kind == Kind::DENSE){
//    std::map<int, int> count;
//    for (int a=0; a <=255; a++){
//      count[a] = 0;
//    }
    Tensor<uint8_t> t{"T" + toStr(i), {w*h,3}, {Dense,Dense}};
    for (int j=0; j<h; j++) {
      for (int k = 0; k < w; k++) {
        for (int c = 0; c < 3; c++) {
//        t(j,k) = image[j*h + k];
//          count[image[j * w * 3 + k * 3 + c]]++;
          t(j * h + k, c) = image[j * w * 3 + k * 3 + c];
        }
      }
    }
//    for (int a=0; a<=255; a++){
//      std::cout << a << "," << count[a] << endl;
//    }
    return {t,w*h*3};
  } else if (kind == Kind::LZ77){
////    return {makeLZ77<uint8_t>("T" + toStr(i),
////                             {(int)height*(int)width},
////                             {0, pos.back()}, compressed),
////            compressed.size()};
//    return {makeLZ77<uint8_t>("T" + toStr(i),
//                              {(int)height*(int)width},
//                              {0, (int)packed_rle.size()}, packed_rle),
//            packed_rle.size()};
//
  } else if (kind == Kind::SPARSE){
    Tensor<uint8_t> t{"T" + toStr(i), {w*h,3}, {Sparse,Dense}, 255};
    for (int j=0; j<h; j++) {
      for (int k = 0; k < w; k++) {
        for (int c = 0; c < 3; c++) {
          if (image[j * w * 3 + k * 3 + c] != 0) {
            t(j * h + k, c) = image[j * w * 3 + k * 3 + c];
          }
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  } else if (kind == Kind::RLE){
    Tensor<uint8_t> t{"T" + toStr(i), {w*h,3}, {RLE_size(4),Dense}, 255};
    uint8_t curr[3] = {image[0], image[1], image[2]};
    t(0,0) = curr[0];
    t(0,1) = curr[1];
    t(0,0) = curr[2];
    for (int j=0; j<h; j++) {
      for (int k = 0; k < w; k++) {
        if (curr[0] != image[j * w * 3 + k * 3 + 0] ||
            curr[1] != image[j * w * 3 + k * 3 + 1] ||
            curr[2] != image[j * w * 3 + k * 3 + 2]){
          curr[0] = image[j * w * 3 + k * 3 + 0];
          curr[1] = image[j * w * 3 + k * 3 + 1];
          curr[2] = image[j * w * 3 + k * 3 + 2];
          for (int c = 0; c < 3; c++) {
            t(j * h + k, c) = image[j * w * 3 + k * 3 + c];
          }
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  }

  Tensor<uint8_t> t{"T" + toStr(i), {3}, {RLE}, 255};
  return {t, 0};
}


uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path){
  std::vector<unsigned char> png_mine;
  auto error = lodepng::encode(png_mine, valsVec, 1111, 1111, LCT_GREY);
  if(!error) lodepng::save_file(png_mine, path);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  return error;
}

std::ostream& operator << (std::ostream& os, const Kind& obj)
{
  switch (obj) {
    case Kind::DENSE: 
      os << "Dense";
      break;
    case Kind::SPARSE: 
      os << "Sparse";
      break;
    case Kind::LZ77: 
      os << "LZ77";
      break;
    case Kind::RLE: 
      os << "RLE";
      break;
  }
  return os;
}

Kernel getKernel(IndexStmt indexStmt, Tensor<uint8_t> t){
  shared_ptr<ir::Module> module = t.getModule();
  void* compute  = module->getFuncPtr("compute");
  void* assemble = module->getFuncPtr("assemble");
  void* evaluate = module->getFuncPtr("evaluate");
  Kernel k(indexStmt, module, evaluate, assemble, compute);
  return k;
}

static void bench_sketch_alpha(benchmark::State& state, Kind kind) {
  int index = state.range(0);

  Kernel kernel;
  auto plus_ = getPlusFunc();
  auto rle_plus = getPlusRleFunc();
  auto copy = getCopyFunc();

  Tensor<uint8_t> dense0 = read_png(index, Kind::DENSE).first;
  Tensor<uint8_t> dense1 = read_png(index+10000, Kind::DENSE).first;
  Tensor<uint8_t> out("out", {1111*1111}, Format{Dense});
  const IndexVar i("i"), j("j");
  IndexStmt indexStmt = (out(i) = plus_(0.5*dense0(i), 0.5*dense1(i)));
  out.compile();
  out.assemble();
  out.compute();

  auto valsArr = out.getStorage().getValues();
  std::vector<uint8_t> vals_expected((uint8_t*)valsArr.getData(),(uint8_t*)valsArr.getData()+valsArr.getSize());
  saveTensor(vals_expected, getValidationOutputPath() + "expected_" + std::to_string(index) + ".png");

  auto res0 = read_png(index, kind);
  auto res1 = read_png(index+10000, kind);
  Tensor<uint8_t> d0 = res0.first;
  Tensor<uint8_t> d1 = res1.first;


  switch (kind){
    case Kind::DENSE:{
      Tensor<uint8_t> denseResult("denseResult", {1111*1111}, Format{Dense});
      IndexStmt indexStmt = (denseResult(i) = plus_(0.5*d0(i), 0.5*d1(i)));
      denseResult.compile();
      denseResult.assemble();

      Kernel k = getKernel(indexStmt, denseResult);
      auto outStorage = denseResult.getStorage();
      auto d0Storage = d0.getStorage();
      auto d1Storage = d1.getStorage();
      taco_tensor_t* a0 = outStorage;
      taco_tensor_t* a1 = d0Storage;
      taco_tensor_t* a2 = d1Storage;

      for (auto _ : state) {
        k.compute(a0,a1,a2);
      }

      k.unpack(3, {a0,a1,a2}, {outStorage, d0Storage, d1Storage});

      // auto arr = outStorage.getValues();
      // std::vector<uint8_t> vals_actual((uint8_t*)arr.getData(),(uint8_t*)arr.getData()+arr.getSize());
      std::vector<uint8_t> vals_actual(a0->vals,a0->vals+(1111*1111));

      if (vals_expected != vals_actual){
        // std::cerr << "[VALIDATION_ERROR]: bench_sketch_alpha, " << kind << ", " << index << ": ";
        // int error_cnt = 0;
        // for (int i=0; i<vals_expected.size(); i++){
        //   if (vals_expected[i] != vals_actual[i]){
        //     std::cerr << i << "(" << (unsigned)vals_expected[i] << "," << (unsigned)vals_actual[i] <<  ")" << ", ";
        //     error_cnt++;
        //   }
        //   if (error_cnt > 10){
        //     std::cerr << ".....";
        //     break;
        //   }
        // }
        // std::cerr << "\b\b" << std::endl;
        saveTensor(vals_actual, getValidationOutputPath() + "dense_" + std::to_string(index) + "_out.png");
      }
      break;
    }
    case Kind::SPARSE:{
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{Sparse}, 255);
      IndexStmt indexStmt = (expected(i) = 0.5*d0(i) + 0.5*d1(i));
      expected.compile();
      expected.assemble();

      Kernel k = getKernel(indexStmt, expected);
      auto outStorage = expected.getStorage();
      auto d0Storage = d0.getStorage();
      auto d1Storage = d1.getStorage();
      taco_tensor_t* a0 = expected.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();

      for (auto _ : state) {
        k.compute(a0,a1,a2);
      }

      k.unpack(3, {a0,a1,a2}, {outStorage, d0Storage, d1Storage});

      Tensor<uint8_t> denseOut("denseOut", {1111*1111}, Format{Dense}, 255);
      denseOut(i) = copy(expected(i));
      denseOut.setAssembleWhileCompute(true);
      denseOut.compile();
      denseOut.compute();

      auto arr = denseOut.getStorage().getValues();
      std::vector<uint8_t> vals_actual((uint8_t*)arr.getData(),(uint8_t*)arr.getData()+arr.getSize());

      if (vals_expected != vals_actual){
        std::cerr << "[VALIDATION_ERROR]: bench_sketch_alpha, " << kind << ", " << index << std::endl;
        saveTensor(vals_actual, getValidationOutputPath() + "sparse_" + std::to_string(index) + "_out.png");
      }

      break;
    }
    case Kind::RLE: {
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{RLE}, 255);
      IndexStmt indexStmt = (expected(i) = rle_plus(0.5*d0(i), 0.5*d1(i)));
      expected.compile();
      expected.assemble();

      Kernel k = getKernel(indexStmt, expected);
      auto outStorage = expected.getStorage();
      auto d0Storage = d0.getStorage();
      auto d1Storage = d1.getStorage();
      taco_tensor_t* a0 = expected.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();

      for (auto _ : state) {
        k.compute(a0,a1,a2);
      }

      k.unpack(3, {a0,a1,a2}, {outStorage, d0Storage, d1Storage});

      Tensor<uint8_t> denseOut("denseOut", {1111*1111}, Format{Dense}, 255);
      denseOut(i) = copy(expected(i));
      denseOut.setAssembleWhileCompute(true);
      denseOut.compile();
      denseOut.compute();

      auto arr = denseOut.getStorage().getValues();
      std::vector<uint8_t> vals_actual((uint8_t*)arr.getData(),(uint8_t*)arr.getData()+arr.getSize());

      if (vals_expected != vals_actual){
        std::cerr << "[VALIDATION_ERROR]: bench_sketch_alpha, " << kind << ", " << index << std::endl;
        saveTensor(vals_actual, getValidationOutputPath() + "rle_" + std::to_string(index) + "_out.png");
      }

      break;
    }
    case Kind::LZ77: {
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{LZ77});
      IndexStmt indexStmt = (expected(i) = plus_(0.5*d0(i), 0.5*d1(i)));
      expected.setAssembleWhileCompute(true);
      expected.compile();

      Kernel k = getKernel(indexStmt, expected);
      auto outStorage = expected.getStorage();
      auto d0Storage = d0.getStorage();
      auto d1Storage = d1.getStorage();
      taco_tensor_t* a0 = expected.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();

      for (auto _ : state) {
        k.compute(a0,a1,a2);
      }

     int numVals = 0; 
     int* pos = (int*)(a0->indices[0][0]);
     std::vector<uint8_t> valsVec(a0->vals,a0->vals+pos[1]);
     std::vector<uint8_t> vals_actual = unpackLZ77_bytes(valsVec, numVals);

      if (vals_expected != vals_actual){
        std::cerr << "[VALIDATION_ERROR]: bench_sketch_alpha, " << kind << ", " << index;// << ": ";
        // for (int i=0; i< vals_expected.size(); i++){
        //   if (vals_expected[i] != vals_actual[i]){
        //     std::cout << i << ", ";
        //   }
        // }
        std::cout << std::endl;
        saveTensor(vals_actual, getValidationOutputPath() + "lz77_" + std::to_string(index) + "_out.png");
      }
      break;
    }
    default:
      return;
  }
}

void sketch_alpha_blending(){
  Kernel kernel;
  auto plus_ = getPlusFunc();
  auto rle_plus = getPlusRleFunc();
  auto copy = getCopyFunc();
  bool time = true;
  taco::util::TimeResults timevalue{};

  int repetitions = getValidationOutputPath() == "" ? 1000 : 1; 

  auto name = "/data/scratch/danielbd/taco-compression-benchmarks/out/ALPH_BLENDING_SKETCH.csv";
  std::ofstream outputFile(name);


  outputFile << "index,isDense,total_bytes,mean,stddev,median" << std::endl;

  for (int index=1; index<=1000; index++){
    std::cout << "INDEX: " << index << std::endl;
    Tensor<uint8_t> denseResult("denseResult", {1111*1111}, Format{Dense});
    {
      Kind kind = Kind::DENSE;
      auto res0 = read_png(index, kind);
      Tensor<uint8_t> d0 = res0.first;
      auto res1 = read_png(index+1000, kind);
      Tensor<uint8_t> d1 = res1.first;
      const IndexVar i("i"), j("j");
      IndexStmt indexStmt = (denseResult(i) = plus_(0.5*d0(i), 0.5*d1(i)));
      denseResult.compile();
      denseResult.assemble();

      Kernel k = getKernel(indexStmt, denseResult);

      auto outStorage = denseResult.getStorage();
      auto d0Storage = d0.getStorage();
      auto d1Storage = d1.getStorage();
      taco_tensor_t* a0 = denseResult.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();
      outputFile << index << "," << "DENSE" << "," << res0.second + res1.second  << ",";
      TOOL_BENCHMARK_REPEAT(
              k.compute(a0,a1,a2),
              "Compute",
              repetitions, outputFile);
      outputFile << std::endl;

      // if (auto valPath = getValidationOutputPath(); valPath != ""){
      //   k.unpack(3, {a0,a1,a2}, {outStorage, d0Storage, d1Storage});
      //   auto vals = outStorage.getValues();
      //   std::vector<uint8_t> valsVec((uint8_t*)vals.getData(),(uint8_t*)vals.getData()+vals.getSize());
      //   saveTensor(valsVec, valPath + "dense_" + std::to_string(index) + ".png");
      // }
    }
    {
      Kind kind = Kind::LZ77;
      auto res0 = read_png(index, kind);
      Tensor<uint8_t> d0 = res0.first;
      auto res1 = read_png(index+1000, kind);
      Tensor<uint8_t> d1 = res1.first;
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{LZ77});
      const IndexVar i("i"), j("j");
      IndexStmt indexStmt = (expected(i) = plus_(0.5*d0(i), 0.5*d1(i)));
      expected.setAssembleWhileCompute(true);
      expected.compile();

      shared_ptr<ir::Module> module = expected.getModule();
      void* compute  = module->getFuncPtr("compute");
      void* assemble = module->getFuncPtr("assemble");
      void* evaluate = module->getFuncPtr("evaluate");

      Kernel k(indexStmt, module, evaluate, assemble, compute);
      taco_tensor_t* a0 = denseResult.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();

      outputFile << index << "," << "LZ77" << "," << res0.second + res1.second << ",";
      TOOL_BENCHMARK_REPEAT(
              k.compute(a0,a1,a2),
              "Compute",
              repetitions, outputFile);
      outputFile << std::endl;

      // if (auto valPath = getValidationOutputPath(); valPath != ""){
      //   int* pos = (int*)(a0->indices[0][0]);
      //   std::vector<uint8_t> valsVec(a0->vals,a0->vals+pos[1]);
      //   int numVals = 0;
      //   saveTensor(unpackLZ77_bytes(valsVec, numVals), valPath + "lz77_" + std::to_string(index) + ".png");
      // }
    }
    {
      Kind kind = Kind::SPARSE;
      auto res0 = read_png(index, kind);
      Tensor<uint8_t> d0 = res0.first;
      auto res1 = read_png(index+1000, kind);
      Tensor<uint8_t> d1 = res1.first;
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{Sparse}, 255);
      const IndexVar i("i"), j("j");
      IndexStmt indexStmt = (expected(i) = 0.5*d0(i) + 0.5*d1(i));
      expected.compile();
      expected.assemble();

      Kernel k = getKernel(indexStmt, expected);
      auto& es = expected.getStorage();
      taco_tensor_t* a0 = es;
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();
      outputFile << index << "," << "SPARSE" << "," << res0.second + res1.second  << ",";
      TOOL_BENCHMARK_REPEAT(
              k.compute(a0,a1,a2),
              "Compute",
              repetitions, outputFile);
      outputFile << std::endl;

      // if (auto valPath = getValidationOutputPath(); valPath != ""){
      //   k.unpack(1, {a0}, {es});
      //   expected.setStorage(es);
      //   std::cout << "after unpack \n";

      //   Tensor<uint8_t> denseOutput("dense", {1111*1111}, Format{Dense}, 255);
      //   denseOutput(i) = expected(i);
      //   denseOutput.setAssembleWhileCompute(true);
      //   denseOutput.compile();
      //   std::cout << "before compute \n";
      //   denseOutput.compute();
      //   std::cout << "after compute \n";

      //   auto vals = denseOutput.getStorage().getValues();
      //   std::vector<uint8_t> valsVec((uint8_t*)vals.getData(),(uint8_t*)vals.getData()+vals.getSize());
      //   saveTensor(valsVec, valPath + "sparse_" + std::to_string(index) + ".png");
      // }
    }
    {
      Kind kind = Kind::RLE;
      auto res0 = read_png(index, kind);
      Tensor<uint8_t> d0 = res0.first;
      auto res1 = read_png(index+1000, kind);
      Tensor<uint8_t> d1 = res1.first;
      Tensor<uint8_t> expected("expected_", {1111*1111}, Format{RLE}, 255);
      const IndexVar i("i"), j("j");
      IndexStmt indexStmt = (expected(i) = rle_plus(0.5*d0(i), 0.5*d1(i)));
      expected.compile();
      expected.assemble();

      shared_ptr<ir::Module> module = expected.getModule();
      void* compute  = module->getFuncPtr("compute");
      void* assemble = module->getFuncPtr("assemble");
      void* evaluate = module->getFuncPtr("evaluate");

      Kernel k(indexStmt, module, evaluate, assemble, compute);
      taco_tensor_t* a0 = expected.getStorage();
      taco_tensor_t* a1 = d0.getStorage();
      taco_tensor_t* a2 = d1.getStorage();

      outputFile << index << "," << "RLE" << "," << res0.second + res1.second  << ",";
      TOOL_BENCHMARK_REPEAT(
              k.compute(a0,a1,a2),
              "Compute",
              repetitions, outputFile);
      outputFile << std::endl;

      // if (auto valPath = getValidationOutputPath(); valPath != ""){
      //   k.unpack(3, {a0,a1,a2}, {expected.getStorage(), d0.getStorage(), d1.getStorage()});

      //   Tensor<uint8_t> denseOutput("dense", {1111*1111}, Format{Dense}, 255);
      //   denseOutput(i) = copy(expected(i));
      //   denseOutput.compile();
      //   denseOutput.assemble();
      //   denseOutput.compute();

      //   auto vals = denseOutput.getStorage().getValues();
      //   std::vector<uint8_t> valsVec((uint8_t*)vals.getData(),(uint8_t*)vals.getData()+vals.getSize());
      //   saveTensor(valsVec, valPath + "rle_" + std::to_string(index) + ".png");
      // }
    }

  }
}



// TACO_BENCH_ARGS(bench_sketch_alpha, sketch_alpha_dense, Kind::DENSE)->DenseRange(1,5,1);
// TACO_BENCH_ARGS(bench_sketch_alpha, sketch_alpha_lz77, Kind::LZ77)->DenseRange(1,5,1);
// TACO_BENCH_ARGS(bench_sketch_alpha, sketch_alpha_sparse, Kind::SPARSE)->DenseRange(1,5,1);
// TACO_BENCH_ARGS(bench_sketch_alpha, sketch_alpha_rle, Kind::RLE)->DenseRange(1,5,1);

void animation_alpha_blending(){
  Kernel kernel;
  auto plus_ = getPlusFunc();
  auto rle_plus = getPlusRleFunc();
  auto copy = getCopyFunc();
  bool time = true;
  taco::util::TimeResults timevalue{};

//  std::cout << "index,isDense,total_bytes,mean,stddev,median" << std::endl;

  std::cout << "index,kind,total_bytes" << std::endl;
  for (int n=1; n<=100; n++){
    auto read = read_rgb_png(n, Kind::DENSE);
    std::cout << n << "," << "DENSE" << "," << read.second << std::endl;

    read = read_rgb_png(n, Kind::SPARSE);
    std::cout << n << "," << "SPARSE" << "," << read.second << std::endl;

    read = read_rgb_png(n, Kind::RLE);
    std::cout << n << "," << "RLE" << "," << read.second << std::endl;


  }

}

// int main(){
// //  sketch_alpha_blending();
//   animation_alpha_blending();
// }