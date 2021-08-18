#include "utils.h"

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

std::pair<Tensor<uint8_t>, size_t> to_tensor_rgb(const std::vector<uint8_t> image, int h, int w,
                                            int index, std::string prefix, Kind kind, int& numVals){
 if (kind == Kind::DENSE){
   auto t = makeDense_3(prefix+"dense_" + std::to_string(index), {h,w,3}, image);
   numVals = h*w*3;
   return {t, h*w*3};
 } else if (kind == Kind::LZ77){
   auto packedr = encode_lz77(image);
   auto packed = packedr.first;
   numVals = packedr.second;
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
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getValues().getSize()*5};
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
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getValues().getSize()*5};
 }
}

std::pair<Tensor<uint8_t>, size_t> to_tensor(const std::vector<uint8_t> image, int h, int w, 
                                             int index, std::string prefix, Kind kind, int& numVals){
  if (kind == Kind::DENSE){
    auto t = makeDense_2(prefix+"dense_" + std::to_string(index), {h,w}, image);
    numVals = h*w;
    return {t, h*w};
  } else if (kind == Kind::LZ77){
    auto pr = encode_lz77(image);
    auto packed = pr.first;
    numVals = pr.second;
    auto t = makeLZ77<uint8_t>(prefix+"lz77_" + std::to_string(index),
                          {h*w},
                          {0, (int)packed.size()}, packed);
    return {t, packed.size()};
  } else if (kind == Kind::SPARSE){
    Tensor<uint8_t> t{prefix+"sparse_" + std::to_string(index), {h,w}, {Dense,Sparse}, 0};
    for (int row=0; row<h; row++){
      for (int col=0; col<w; col++){
        if (image[row*w + col] != 0){
            t(row,col) = image[row*w + col];
        }
      }
    }
    t.pack();
    numVals = t.getStorage().getValues().getSize();
    return {t, t.getStorage().getValues().getSize()*5};
  } else if (kind == Kind::RLE){
    Tensor<uint8_t> t{prefix+"rle_" + std::to_string(index), {h,w}, {Dense,RLE}, 0};
    uint8_t curr = image[0];
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
    return {t, t.getStorage().getValues().getSize()*5};
  }
}

std::pair<Tensor<uint8_t>, size_t> read_movie_frame(std::string img_folder, std::string prefix, int index, Kind kind, int& w, int& h, int& numVals) {
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

 return to_tensor_rgb(image,h,w,index,prefix+"f" + std::to_string(index) + "_", kind, numVals);
}

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path, int width, int height){
  std::vector<unsigned char> png_mine;
  auto error = lodepng::encode(png_mine, valsVec, width, height, LCT_GREY);
  if(!error) lodepng::save_file(png_mine, path);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  return error;
}

uint32_t saveTensor_RGB(std::vector<unsigned char> valsVec, std::string path, int width, int height){
  std::vector<unsigned char> png_mine;
  auto error = lodepng::encode(png_mine, valsVec, width, height, LCT_RGB);
  if(!error) lodepng::save_file(png_mine, path);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  return error;
}

void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, bool isroi, std::string bench_kind, int index, std::string prefix){
 const IndexVar i("i"), j("j"), c("c");
 auto copy = getCopyFunc();
 auto dims = roi_t.getDimensions();

 Tensor<uint8_t> v("v", dims, dims.size() == 1? Format{Dense} : Format{Dense,Dense,Dense});
 if (kind == Kind::LZ77){
   v(i) = copy((isroi ? 255 : 1) * roi_t(i));
 } else {
   v(i,j,c) = copy((isroi ? 255 : 1) * roi_t(i,j,c));
 }
 v.setAssembleWhileCompute(true);
 v.compile();
 v.compute();

 uint8_t* start = (uint8_t*) v.getStorage().getValues().getData();
 std::vector<uint8_t> validation(start, start+w*h*3);
 saveTensor_RGB(validation, getValidationOutputPath() + prefix + "_" + bench_kind+ "_" + std::to_string(index) + ".png",  w, h);
}

void saveValidation(Tensor<uint8_t> roi_t, Kind kind, int w, int h, std::string bench_kind, int index, std::string prefix, bool is_roi){
  const IndexVar i("i"), j("j");
  auto copy = getCopyFunc();
  auto dims = roi_t.getDimensions();

  Tensor<uint8_t> v("v", dims, dims.size() == 1? Format{Dense} : Format{Dense,Dense});
  if (kind == Kind::LZ77){
    v(i) = copy((is_roi ? 255 : 1)*roi_t(i));
  } else {
    v(i,j) = copy((is_roi ? 255 : 1)*roi_t(i,j));
  }
  v.setAssembleWhileCompute(true);
  v.compile();
  v.compute();

  uint8_t* start = (uint8_t*) v.getStorage().getValues().getData();
  std::vector<uint8_t> validation(start, start+w*h);
  saveTensor(validation, getValidationOutputPath() + prefix + "_" + bench_kind+ "_" + std::to_string(index) + ".png",  w, h);
}

std::pair<int,int> count_bytes_vals(Tensor<uint8_t> t, Kind kind){
  if (kind == Kind::DENSE){
    int num = t.getStorage().getValues().getSize();
    return {num, num};
  } else if (kind == Kind::LZ77){
    int numVals = 0;
    int numBytes = t.getStorage().getValues().getSize();
    uint8_t* raw_bytes = (uint8_t*) t.getStorage().getValues().getData();
    std::vector raw(raw_bytes, raw_bytes + numBytes);
    unpackLZ77_bytes(raw, numVals);
    return {numBytes, numVals};
  } else if (kind == Kind::SPARSE){
    return {t.getStorage().getValues().getSize()*5, t.getStorage().getValues().getSize()};
  } else if (kind == Kind::RLE){
    return {t.getStorage().getValues().getSize()*5, t.getStorage().getValues().getSize()};
  }
  return {-1,-1};
}

std::string to_string(Kind k){
  switch (k){
    case Kind::DENSE: return "DENSE";
    case Kind::SPARSE: return "SPARSE";
    case Kind::RLE: return "RLE";
    case Kind::LZ77: return "LZ77";
  }
  return "";
}
