#include "png_reader.h"
#include "../test/test.h"
#include "taco/util/timers.h"
#include "bench.h"
#include "lodepng.h"
#include "utils.h"

#include <iostream>
#include <random>
#include <variant>
#include <climits>
#include <limits>

Func getBrightenFunc(uint8_t brightness, bool full){
  auto brighten = [=](const std::vector<ir::Expr>& v) {
      auto sum = ir::Add::make(v[0], brightness);
      return ternaryOp(ir::Gt::make(sum, 255), 255, sum);
  };
  auto algFunc = [=](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      if (full){
        return IterationAlgebra(Union(l, Complement(l)));
      } else {
        return IterationAlgebra(l);
      }
  };
  Func plus_("plus_", brighten, algFunc);
  return plus_;
}

std::vector<uint8_t> raw_image(std::string filename, int& w, int& h){
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

std::pair<std::vector<Tensor<uint8_t>>, size_t> read_rgb_sequence(int start, int end, Kind kind) {
  auto img_folder = getEnvVar("IMAGE_FOLDER");
  if (img_folder == "") {
    img_folder = "/Users/danieldonenfeld/Developer/png_analysis/bs_png/";
  }

  std::vector<Tensor<uint8_t>> tensors;
  tensors.reserve(end - start + 1);
  size_t total_size = 0;

  int w = 0;
  int h = 0;
  int l = 0;
  for (int i=start; i<=end; i++){
    std::ostringstream stringStream;
    stringStream << img_folder;
    if (i < 10){
      stringStream << "00";
    } else if (i < 100){
      stringStream << "0";
    }
    stringStream << i << ".png";
    std::string filename = stringStream.str();
    auto image = raw_image(filename, w, h);

    if (kind == Kind::DENSE){
      auto t = makeDense_3("dense_" + std::to_string(start) + "_" + std::to_string(end), {h,w,3}, image);
      total_size += w*h*3;
      tensors.push_back(t);
    } else if (kind == Kind::LZ77){
      auto packed = encode_lz77(image).first;
      auto t = makeLZ77<uint8_t>("lz77_" + std::to_string(start) + "_" + std::to_string(end),
                            {h*w*3},
                            {0, (int)packed.size()}, packed);
      tensors.push_back(t);
      total_size+=packed.size();
    } else if (kind == Kind::SPARSE){
      Tensor<uint8_t> t{"sparse_" + std::to_string(start) + "_" + std::to_string(end), {h,w,3}, {Dense,Sparse,Dense}, 255};
        for (int row=0; row<h; row++){
          for (int col=0; col<w; col++){
            if (image[row*w*3 + col*3 + 0] != 255 ||
                image[row*w*3 + col*3 + 1] != 255 ||
                image[row*w*3 + col*3 + 2] != 255){
              for (int color=0; color<3; color++){
                t(row,col,color) = image[row*w*3 + col*3 + color];
              }
            }
          }
      }
      t.pack();
      tensors.push_back(t);
      total_size+=t.getStorage().getValues().getSize();
    } else if (kind == Kind::RLE){
      Tensor<uint8_t> t{"rle_" + std::to_string(start) + "_" + std::to_string(end), {h,w,3}, {Dense,RLE_size(3),Dense}, 255};
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
      tensors.push_back(t);
      total_size+=t.getStorage().getValues().getSize();
    }
  }

  return {tensors, total_size};
}

uint32_t saveRGBTensor(std::vector<unsigned char> valsVec, std::string path){
  std::vector<unsigned char> png_mine;
  auto error = lodepng::encode(png_mine, valsVec, 1920, 1080, LCT_RGB);
  if(!error) lodepng::save_file(png_mine, path);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  return error;
}

void brighten_bench(){
  Kernel kernel;
  auto plus_ = getPlusFunc();
  auto rle_plus = getPlusRleFunc();
  auto copy = getCopyFunc();
  bool time = true;
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j"), f("f"), c("c");

  int repetitions = getNumRepetitions(100);

  std::cout << "start,end,kind,total_bytes,mean,stddev,median" << std::endl;

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

  auto bench_kind = getEnvVar("BENCH_KIND");
  
  if (bench_kind == "DENSE") {
    Kind kind = Kind::DENSE;
    auto res = read_rgb_sequence(start, end, Kind::DENSE);
    std::vector<Tensor<uint8_t>> in = res.first;
    std::vector<taco_tensor_t*> outv;
    std::vector<taco_tensor_t*> inv;
    Kernel k;
    for(auto& t : in){
      Tensor<uint8_t> out("out", t.getDimensions(), Format{Dense,Dense,Dense});
      auto brighten = getBrightenFunc(20,true);
      IndexStmt stmt = (out(i,j,c) = brighten(t(i,j,c)));
      out.compile();
      out.assemble();
      // out.compute();
      outv.push_back(out.getTacoTensorT());
      inv.push_back(t.getTacoTensorT());

      k = getKernel(stmt, out);
    }

    std::cout << start << "," << end << "," << "DENSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT({
      for (int f=0; f< numFrames; f++){
        k.compute(outv[f],inv[f]);
      }
    }, "Compute", repetitions, std::cout);
  } else if (bench_kind == "SPARSE") {
    auto res = read_rgb_sequence(start, end, Kind::SPARSE);
    std::vector<Tensor<uint8_t>> in = res.first;
    std::vector<taco_tensor_t*> outv;
    std::vector<taco_tensor_t*> inv;
    Kernel k;
    for(auto& t : in){
      Tensor<uint8_t> out("out", t.getDimensions(), Format{Dense,Sparse,Dense});
      auto brighten = getBrightenFunc(20,false);
      IndexStmt stmt = (out(i,j,c) = brighten(t(i,j,c)));
      out.compile();
      out.assemble();
      // out.compute();
      outv.push_back(out.getTacoTensorT());
      inv.push_back(t.getTacoTensorT());

      k = getKernel(stmt, out);
    }

    std::cout << start << "," << end << "," << "SPARSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT({
      for (int f=0; f< numFrames; f++){
        k.compute(outv[f],inv[f]);
      }
    }, "Compute", repetitions, std::cout);
  } else if (bench_kind == "RLE") {
    auto res = read_rgb_sequence(start, end, Kind::RLE);
    std::vector<Tensor<uint8_t>> in = res.first;
    std::vector<taco_tensor_t*> outv;
    std::vector<taco_tensor_t*> inv;
    Kernel k;
    for(auto& t : in){
      Tensor<uint8_t> out("out", t.getDimensions(), Format{Dense,RLE_size(3),Dense});
      auto brighten = getBrightenFunc(20,false);
      IndexStmt stmt = (out(i,j,c) = brighten(t(i,j,c)));
      out.compile();
      out.assemble();
      // out.compute();
      outv.push_back(out.getTacoTensorT());
      inv.push_back(t.getTacoTensorT());

      k = getKernel(stmt, out);
    }

    std::cout << start << "," << end << "," << "RLE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT({
      for (int f=0; f< numFrames; f++){
        k.compute(outv[f],inv[f]);
      }
    }, "Compute", repetitions, std::cout);
  } else if (bench_kind == "LZ77") {
    auto res = read_rgb_sequence(start, end, Kind::LZ77);
    std::vector<Tensor<uint8_t>> in = res.first;
    std::vector<taco_tensor_t*> outv;
    std::vector<taco_tensor_t*> inv;
    Kernel k;
    for(auto& t : in){
      Tensor<uint8_t> out("out", t.getDimensions(), Format{LZ77});
      auto brighten = getBrightenFunc(20,true);
      IndexStmt stmt = (out(i) = brighten(t(i)));
      out.setAssembleWhileCompute(true);
      out.compile();
      // out.compute();
      outv.push_back(out.getTacoTensorT());
      inv.push_back(t.getTacoTensorT());

      k = getKernel(stmt, out);
    }

    std::cout << start << "," << end << "," << "LZ77" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT({
      for (int f=0; f< numFrames; f++){
        k.compute(outv[f],inv[f]);
      }
    }, "Compute", repetitions, std::cout);
  } else {
    std::cout << "benchmark kind " << bench_kind << " unknown" << std::endl;
  }
}