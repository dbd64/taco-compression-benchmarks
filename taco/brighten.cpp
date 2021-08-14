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

Index makeDenseIndex(int s0, int s1, int s2, int s3) {
  return Index(CSR, {ModeIndex({makeArray({s0})}),
                     ModeIndex({makeArray({s1})}),
                     ModeIndex({makeArray({s2})}),
                     ModeIndex({makeArray({s3})})});
}

template<typename T>
TensorBase makeDense(const std::string& name, const std::vector<int>& dims,
                   const std::vector<T>& vals) {
  Tensor<T> tensor(name, dims, Format{Dense, Dense, Dense, Dense});
  auto storage = tensor.getStorage();
  storage.setIndex(makeDenseIndex(dims[0], dims[1], dims[2], dims[3]));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

ir::Expr ternaryOp(const ir::Expr& c, const ir::Expr& a, const ir::Expr& b){
  // c ? a : b
  ir::Expr a_b = ir::BinOp::make(a,b, " : ");
  return ir::BinOp::make(c, a_b, "(", " ? ", ")");
}

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

std::vector<uint8_t> encode_lz77(const std::vector<uint8_t> in);

std::pair<Tensor<uint8_t>, size_t> read_rgb_sequence(int start, int end, Kind kind) {
  auto img_folder = getEnvVar("IMAGE_FOLDER");
  if (img_folder == "") {
    img_folder = "/Users/danieldonenfeld/Developer/png_analysis/bs_png/";
  }


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
    stringStream << img_folder;
    if (i < 10){
      stringStream << "00";
    } else if (i < 100){
      stringStream << "0";
    }
    stringStream << i << ".png";
    std::string filename = stringStream.str();

    //load and decode
    unsigned error = lodepng::load_file(png, filename);
    // auto state = lodepng::State();
    // if (!error) lodepng::decode(image, width, height, state, png);
    if(!error) error = decode(image, compressed, pos, width, height, png, LCT_RGB);

    //if there's an error, display it
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    
    w = (int)width;
    h = (int)height;
    l++;
    vals.insert(vals.end(), image.begin(), image.end());
  }

  if (kind == Kind::DENSE){
    return {makeDense("dense_" + std::to_string(start) + "_" + std::to_string(end), {l,h,w,3}, vals), l*w*h*3};
  } else if (kind == Kind::LZ77){
    auto packed = encode_lz77(vals);
    return {makeLZ77<uint8_t>("lz77_" + std::to_string(start) + "_" + std::to_string(end),
                          {l*h*w*3},
                          {0, (int)packed.size()}, packed),
        packed.size()};
  } else if (kind == Kind::SPARSE){
    Tensor<uint8_t> t{"sparse_" + std::to_string(start) + "_" + std::to_string(end), {l,h,w,3}, {Dense,Dense,Sparse,Dense}, 255};
    for (int frame=0; frame<l; frame++){
      // std::cout << frame << std::endl;
      for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
          if (vals[frame*w*h*3 + row*w*3 + col*3 + 0] != 255 ||
              vals[frame*w*h*3 + row*w*3 + col*3 + 1] != 255 ||
              vals[frame*w*h*3 + row*w*3 + col*3 + 2] != 255){
            for (int color=0; color<3; color++){
              t(frame,row,col,color) = vals[frame*h*w*3 + row*w*3 + col*3 + color];
            }
          }
        }
      }
    }
    t.pack();
    return {t,t.getStorage().getValues().getSize()};
  } else if (kind == Kind::RLE){
    Tensor<uint8_t> t{"rle_" + std::to_string(start) + "_" + std::to_string(end), {l,h,w,3}, {Dense,Dense,RLE_size(3),Dense}, 255};
    uint8_t curr[3] = {vals[0], vals[1], vals[2]};
    t(0,0,0,0) = curr[0];
    t(0,0,0,1) = curr[1];
    t(0,0,0,2) = curr[2];
    for (int frame=0; frame<l; frame++){
      // std::cout << frame << std::endl;
      for (int row=0; row<h; row++){
        for (int col=0; col<w; col++){
          if (vals[frame*w*h*3 + row*w*3 + col*3 + 0] != curr[0] ||
              vals[frame*w*h*3 + row*w*3 + col*3 + 1] != curr[1] ||
              vals[frame*w*h*3 + row*w*3 + col*3 + 2] != curr[2]){
            curr[0] = vals[frame*w*h*3 + row*w*3 + col*3 + 0];
            curr[1] = vals[frame*w*h*3 + row*w*3 + col*3 + 1];
            curr[2] = vals[frame*w*h*3 + row*w*3 + col*3 + 2];
            for (int color=0; color<3; color++){
              t(frame,row,col,color) = vals[frame*w*h*3 + row*w*3 + col*3 + color];
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

  int repetitions = 100; 

  std::cout << "start,end,kind,total_bytes,mean,stddev,median" << std::endl;

  auto start_str = getEnvVar("IMAGE_START");
  if (start_str == "") {
    return;
  }
  auto end_str = getEnvVar("IMAGE_END");
  if (end_str == "") {
    return;
  }

  int start = std::stoi(start_str);
  int end = std::stoi(end_str);
  int numFrames = end - (start-1);

  {
    Kind kind = Kind::DENSE;
    auto res = read_rgb_sequence(start, end, Kind::DENSE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {numFrames, 1080, 1920, 3}, Format{Dense,Dense,Dense,Dense});
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

    uint8_t* a0vals = a0->vals;
    for (int frame=0; frame < numFrames; frame++){
      std::vector<uint8_t> vals(a0vals,a0vals+1080*1920*3);
      a0vals+=1080*1920*3;
      saveRGBTensor(vals, getValidationOutputPath() + "dense_output_" + std::to_string(frame+start) + ".png");
    }
    uint8_t* a1vals = a1->vals;
    for (int frame=0; frame < numFrames; frame++){
      std::vector<uint8_t> vals(a1vals,a1vals+1080*1920*3);
      a1vals+=1080*1920*3;
      saveRGBTensor(vals, getValidationOutputPath() + "input_" + std::to_string(frame+start) + ".png");
    }
  }
  {
    auto res = read_rgb_sequence(start, end, Kind::SPARSE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {numFrames, 1080, 1920, 3}, Format{Dense,Dense,Sparse,Dense});
    auto brighten = getBrightenFunc(20,false);
    IndexStmt stmt = (out(f,i,j,c) = brighten(in(f,i,j,c)));
    out.compile();
    out.assemble();
    // out.printComputeIR(std::cout);

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "SPARSE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);

    out.compute();
    Tensor<uint8_t> dense("dense", {numFrames, 1080, 1920, 3}, Format{Dense,Dense,Dense,Dense});
    dense(f, i, j, c) = copy(out(f,i,j,c));
    dense.setAssembleWhileCompute(true);
    dense.compile();
    dense.compute();
    // dense.printComputeIR(std::cout);

    taco_tensor_t* denseOut = dense.getStorage();
    uint8_t* a0vals = denseOut->vals;
    for (int frame=0; frame < numFrames; frame++){
      std::vector<uint8_t> vals(a0vals,a0vals+1080*1920*3);
      a0vals+=1080*1920*3;
      saveRGBTensor(vals, getValidationOutputPath() + "sparse_output_" + std::to_string(frame+start) + ".png");
    }
  }
  {
    auto res = read_rgb_sequence(start, end, Kind::RLE);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {numFrames, 1080, 1920, 3}, Format{Dense,Dense,RLE_size(3),Dense});
    auto brighten = getBrightenFunc(20,false);
    IndexStmt stmt = (out(f,i,j,c) = brighten(in(f,i,j,c)));
    out.compile();
    out.assemble();

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "RLE" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);

    out.compute();
    Tensor<uint8_t> dense("dense", {numFrames, 1080, 1920, 3}, Format{Dense,Dense,Dense,Dense});
    dense(f, i, j, c) = copy(out(f,i,j,c));
    dense.setAssembleWhileCompute(true);
    dense.compile();
    dense.compute();

    taco_tensor_t* denseOut = dense.getStorage();
    uint8_t* a0vals = denseOut->vals;
    for (int frame=0; frame < numFrames; frame++){
      std::vector<uint8_t> vals(a0vals,a0vals+1080*1920*3);
      a0vals+=1080*1920*3;
      saveRGBTensor(vals, getValidationOutputPath() + "rle_output_" + std::to_string(frame+start) + ".png");
    }
  }
  {
    auto res = read_rgb_sequence(start, end, Kind::LZ77);
    Tensor<uint8_t> in = res.first;
    Tensor<uint8_t> out("out", {numFrames*1080*1920*3}, Format{LZ77});
    auto brighten = getBrightenFunc(20,true);
    IndexStmt stmt = (out(i) = brighten(in(i)));
    out.setAssembleWhileCompute(true);
    out.compile();

    Kernel k = getKernel(stmt, out);

    auto outStorage = out.getStorage();
    auto inStorage = in.getStorage();
    taco_tensor_t* a0 = outStorage;
    taco_tensor_t* a1 = inStorage;
    std::cout << start << "," << end << "," << "LZ77" << "," << res.second  << ",";
    TOOL_BENCHMARK_REPEAT(
            k.compute(a0,a1),
            "Compute",
            repetitions);

    std::vector<uint8_t> lz77_bytes(a0->vals,a0->vals+((int*)a0->indices[0][0])[1]);
    auto d = unpackLZ77_bytes(lz77_bytes);
    auto a0vals = &d[0];

    for (int frame=0; frame < numFrames; frame++){
      std::vector<uint8_t> vals(a0vals,a0vals+1080*1920*3);
      a0vals+=1080*1920*3;
      saveRGBTensor(vals, getValidationOutputPath() + "lz77_output_" + std::to_string(frame+start) + ".png");
    }
  }
}