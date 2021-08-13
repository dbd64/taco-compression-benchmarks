#include "png_reader.h"
#include "../test/test.h"
#include "taco/util/timers.h"
#include "bench.h"
#include "lodepng.h"

#include <iostream>
#include <random>
#include <variant>
#include <climits>
#include <limits>


ir::Expr ternaryOp(const ir::Expr& c, const ir::Expr& a, const ir::Expr& b){
  // c ? a : b
  ir::Expr a_b = ir::BinOp::make(a,b, " : ");
  return ir::BinOp::make(c, a_b, "(", " ? ", ")");
}

Func getBrightenFunc(uint8_t brightness, bool full){
  auto brighten = [&](const std::vector<ir::Expr>& v) {
      auto sum = ir::Add::make(v[0], brightness);
      return ternaryOp(ir::Gt::make(sum, 255), 255, sum);
  };
  auto algFunc = [&](const std::vector<IndexExpr>& v) {
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

std::pair<Tensor<uint8_t>, size_t> read_rgb_sequence(int start, int end, Kind kind) {
  int w = 0;
  int h = 0;
  int l = 0;
  std::vector<uint8_t> vals;
  for (int i=start; i<=end; i++){
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

    int w = (int)width;
    int h = (int)height;
    l++;
    vals.insert(vals.end(), image.begin(), image.end());
  }

  if (kind == Kind::DENSE){
    Tensor<uint8_t> t{"dense_" + std::to_string(start) + "_" + std::to_string(end), {l,w,h,3}, {Dense,Dense,Dense,Dense}};
    for (int frame=0; frame<l; frame++){
      for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
          for (int color=0; color<3; color++){
            t(frame,row,col,color) = vals[frame*w*h*3 + row*h*3 + col*3 + color];
          }
        }
      }
    }
    return {t,l*w*h*3};
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
    Tensor<uint8_t> t{"sparse_" + std::to_string(start) + "_" + std::to_string(end), {l,w,h,3}, {Dense,Dense,Sparse,Dense}, 0};
    for (int frame=0; frame<l; frame++){
      for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
          if (vals[frame*w*h*3 + row*h*3 + col*3 + 0] != 0 ||
              vals[frame*w*h*3 + row*h*3 + col*3 + 1] != 0 ||
              vals[frame*w*h*3 + row*h*3 + col*3 + 2] != 0){
            for (int color=0; color<3; color++){
              t(frame,row,col,color) = vals[frame*w*h*3 + row*h*3 + col*3 + color];
            }
          }
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  } else if (kind == Kind::RLE){
    Tensor<uint8_t> t{"rle_" + std::to_string(start) + "_" + std::to_string(end), {l,w,h,3}, {Dense,Dense,RLE_size(3),Dense}, 255};
    uint8_t curr[3] = {vals[0], vals[1], vals[2]};
    t(0,0,0,0) = curr[0];
    t(0,0,0,1) = curr[1];
    t(0,0,0,2) = curr[2];
    for (int frame=0; frame<l; frame++){
      for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
          if (vals[frame*w*h*3 + row*h*3 + col*3 + 0] != curr[0] ||
              vals[frame*w*h*3 + row*h*3 + col*3 + 1] != curr[1] ||
              vals[frame*w*h*3 + row*h*3 + col*3 + 2] != curr[2]){
            curr[0] = vals[frame*w*h*3 + row*h*3 + col*3 + 0];
            curr[1] = vals[frame*w*h*3 + row*h*3 + col*3 + 1];
            curr[2] = vals[frame*w*h*3 + row*h*3 + col*3 + 2];
            for (int color=0; color<3; color++){
              t(frame,row,col,color) = vals[frame*w*h*3 + row*h*3 + col*3 + color];
            }
          }
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  }

  Tensor<uint8_t> t{"error", {1}, {Dense}, 255};
  return {t, 0};
}


void brighten_bench(){
  Kernel kernel;
  auto plus_ = getPlusFunc();
  auto rle_plus = getPlusRleFunc();
  auto copy = getCopyFunc();
  bool time = true;
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j"), f("f"), c("c");

  int repetitions = getValidationOutputPath() == "" ? 1 : 1; 

  std::cout << "start,end,kind,total_bytes,mean,stddev,median" << std::endl;
  int start = 1;
  int end = 1000;

  {
    Kind kind = Kind::DENSE;
    auto res = read_rgb_sequence(start, end, Kind::DENSE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {1000, 1080, 1920, 3}, Format{Dense,Dense,Dense,Dense});
    auto brighten = getBrightenFunc(20,true);
    IndexStmt stmt = (out(f,i,j,c) = brighten(in(f,i,j,c)));
    out.compile();
    out.assemble();

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "DENSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);
  }
  {
    auto res = read_rgb_sequence(start, end, Kind::SPARSE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {1000, 1080, 1920, 3}, Format{Dense,Dense,Sparse,Dense});
    auto brighten = getBrightenFunc(20,true);
    IndexStmt stmt = (out(f,i,j,c) = brighten(in(f,i,j,c)));
    out.compile();
    out.assemble();

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "DENSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);
  }
  {
    auto res = read_rgb_sequence(start, end, Kind::RLE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {1000, 1080, 1920, 3}, Format{Dense,Dense,RLE_size(3),Dense});
    auto brighten = getBrightenFunc(20,true);
    IndexStmt stmt = (out(f,i,j,c) = brighten(in(f,i,j,c)));
    out.compile();
    out.assemble();

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "DENSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);
  }

}