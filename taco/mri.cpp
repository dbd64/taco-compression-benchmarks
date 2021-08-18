#include "bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"
#include "taco/util/timers.h"
#include "utils.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"
#include "png_reader.h"

using namespace taco;


std::vector<uint8_t> raw_image_(std::string filename, int& w, int& h){
    std::vector<unsigned char> png;
    std::vector<unsigned char> image; //the raw pixels
    std::vector<unsigned char> compressed;
    std::vector<int> pos;
    unsigned width = 0, height = 0;

    unsigned error = lodepng::load_file(png, filename);
    if(!error) error = decode(image, compressed, pos, width, height, png, LCT_GREY);

    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    
    w = (int)width;
    h = (int)height;
    return std::move(image);
}


std::pair<Tensor<uint8_t>, size_t> read_mri_image(int index, int threshold, Kind kind, int& w, int& h, int& numVals) {
  auto img_folder = getEnvVar("IMAGE_FOLDER");
  if (img_folder == "") {
    img_folder = "/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data/mri/";
  }

  std::ostringstream stringStream;
  stringStream << img_folder << index << ".png";
  std::string filename = stringStream.str();
  auto image = raw_image_(filename, w, h);

  // Threshold image
  for (auto& i : image){
    i = i > threshold ? 1 : 0; 
  }

  return to_tensor(image,h,w,index,"_mri_" + std::to_string(threshold) + "_", kind, numVals);
}


std::pair<Tensor<uint8_t>, size_t> generateROI(int width, int height, int index, Kind kind, int& numVals) {
  vector<uint8_t> vals(width*height, 0);
  if (width <100 || height < 100) {
    std::cout << "SIZE ERROR: " << width << ", " << height << std::endl;
  }
  // ROIs each 40x40
  int w2 = width/2;
  int h2 = height/2;
  std::vector<std::pair<int,int>> rois{{-50,-50}, {10,-50}, {-50,10}, {10,10}};
  for (auto& roi : rois){
    int row = roi.second + h2; 
    int col = roi.first + w2; 
    for (int r=row; r<row+40; r++){
      for (int c=col; c<col+40; c++){
        if (r*width + c >= vals.size()){
          std::cout << "ERROR " << r << ", " << c << ": " << width << " " << height << " " << row << " " << col << std::endl;
        } else {
          vals[r*width + c] = 1;
        }
      }
    }
  }
  return to_tensor(vals, height, width, index, "roi_", kind, numVals);
}


struct XorOp {
    ir::Expr xorf(ir::Expr l, ir::Expr r){
        return ir::BinOp::make(l,r, "(", "^", ")");
    }

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 3) << "Requires 3 arguments (img_t1, img_t2, ROI) ";
        return xorf(v[0], v[1]);
    }
};

struct AndOp {
    ir::Expr andf(ir::Expr l, ir::Expr r){
        return ir::BinOp::make(l,r, "(", "&", ")");
    }

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 3) << "Requires 3 arguments (img_t1, img_t2, ROI) ";
        return andf(v[0], v[1]);
    }
};

struct XorAndOp {
  ir::Expr xorf(ir::Expr l, ir::Expr r){
    return ir::BinOp::make(l,r, "(", "^", ")");
  }

  ir::Expr andf(ir::Expr l, ir::Expr r){
    return ir::BinOp::make(l,r, "(", "&", ")");
  }

  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 3) << "Requires 3 arguments (img_t1, img_t2, ROI) ";
    return xorf(andf(v[0], v[2]), andf(v[1], v[2]));
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

struct universeAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    auto t = Union(regions[0], Complement(regions[0]));
    for (int i=1; i< regions.size(); i++){
      t = Union(t, Union(regions[i], Complement(regions[i])));
    }
    return t;
  }
}; 

struct xorAndAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    auto m1 = Intersect(regions[0], regions[2]);
    auto m2 = Intersect(regions[1], regions[2]);
    auto noIntersect = Complement(Intersect(Intersect(regions[0], regions[1]), regions[2]));
    return Intersect(noIntersect, Union(m1, m2));
  }
};

struct Boolean {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 1) << "Add operator needs at least one operand";
    return ir::Literal::make(int64_t(1), v[0].type());
  }
};

Func xorAndOp("fused_xor_and", XorAndOp(), unionAlgebra());
Func xorOp_lz("xor", XorOp(), universeAlgebra());
Func andOp_lz("and", AndOp(), universeAlgebra());
Func xorAndOp_lz("fused_xor_and", XorAndOp(), universeAlgebra());
Func xorAndOp_sparse("fused_xor_and", Boolean(), xorAndAlgebra());

void writeHeader(std::ostream& os, int repetitions){
  os << "index,kind,total_vals,total_bytes,mean,stddev,median,";
  for (int i=0; i<repetitions-1; i++){
    os << i << ","; 
  }
  os << repetitions-1;
}

void bench(std::string bench_kind){
  bool time = true;
  auto copy = getCopyFunc();
  taco::util::TimeResults timevalue{};
  const IndexVar i("i"), j("j");

  int repetitions = 100;
  

  Kind kind;
  Format f;
  if (bench_kind == "DENSE") {
    kind = Kind::DENSE;
    f = Format{Dense,Dense};
  } else if (bench_kind == "SPARSE"){
    kind = Kind::SPARSE;
    f = Format{Dense,Sparse};
  } else if (bench_kind == "RLE"){
    kind = Kind::RLE;
    f = Format{Dense,RLE};
  } else if (bench_kind == "LZ77"){
    kind = Kind::LZ77;
    f = Format{LZ77};
  }

  std::ofstream outputFile(getOutputPath() + to_string(kind) + "_mri.csv");
  writeHeader(outputFile, repetitions);

  for (int index=1; index<=253; index++){
    int w = 0;
    int h = 0;
    int t1_num_vals = 0;
    int t2_num_vals = 0;
    int roi_num_vals = 0;
    auto img_t1_res =  read_mri_image(index, (int) (0.75*255), kind, w, h, t1_num_vals);
    auto img_t2_res =  read_mri_image(index, (int) (0.80*255), kind, w, h, t2_num_vals);
    auto dims = img_t2_res.first.getDimensions();
    auto roi = generateROI(w,h, index, kind, roi_num_vals);

    auto total_vals = t1_num_vals + t2_num_vals + roi_num_vals;
    auto total_bytes = img_t1_res.second + img_t2_res.second + roi.second;

    auto img_t1 = img_t1_res.first;
    auto img_t2 = img_t2_res.first;
    auto roi_t = roi.first;

    Tensor<uint8_t> out("out", dims, f);
    IndexStmt stmt;
    if (kind == Kind::LZ77){
      stmt = (out(i) = xorAndOp_lz(img_t1_res.first(i), img_t2_res.first(i), roi.first(i)));
    } else if (kind == Kind::SPARSE) {
      stmt = (out(i,j) = xorAndOp_sparse(img_t1_res.first(i,j), img_t2_res.first(i,j), roi.first(i,j)));
    } else {
      stmt = (out(i,j) = xorAndOp(img_t1_res.first(i,j), img_t2_res.first(i,j), roi.first(i,j)));
    }
    out.setAssembleWhileCompute(true);
    out.compile();
    Kernel k = getKernel(stmt, out);

    taco_tensor_t* a0 = out.getStorage();
    taco_tensor_t* a1 = img_t1.getStorage();
    taco_tensor_t* a2 = img_t2.getStorage();
    taco_tensor_t* a3 = roi_t.getStorage();

    std::cout << index << "," << bench_kind << "," << total_vals << "," << total_bytes << ",";
    TOOL_BENCHMARK_REPEAT({
        k.compute(a0,a1,a2,a3);
    }, "Compute", repetitions, outputFile);
    outputFile << std::endl;

    out.compute();

    // saveValidation(img_t1, kind, w, h, bench_kind, index, "img_t1", true);
    // saveValidation(img_t2, kind, w, h, bench_kind, index, "img_t2", true);
    // saveValidation(roi_t, kind, w, h, bench_kind, index, "roi", true);
    // saveValidation(out, kind, w, h, bench_kind, index, "out", true);
  }
}

void mri_bench(){
  auto bench_kind = getEnvVar("BENCH_KIND");
  bench(bench_kind);
}