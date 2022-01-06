#ifndef PNG_READER_H
#define PNG_READER_H

#include "taco.h"
#include "taco/tensor.h"
#include "lodepng.h"
#include <iterator>
#include <variant>

using namespace taco;

#define TACO_TIME_REPEAT(CODE, REPEAT, RES, COLD) {  \
    taco::util::Timer timer;                         \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
    }                                                \
    RES = timer.getResult();                         \
  }

#define TOOL_BENCHMARK_REPEAT(CODE, NAME, REPEAT, STREAM) {      \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,REPEAT,timevalue,true);              \
      STREAM << timevalue;                                       \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_REPEAT_WARM(CODE, NAME, REPEAT, STREAM) { \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,REPEAT,timevalue,false);             \
      STREAM << timevalue;                                       \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_TIMER(CODE,NAME,TIMER) {                  \
    if (time) {                                                  \
      taco::util::Timer timer;                                   \
      timer.start();                                             \
      CODE;                                                      \
      timer.stop();                                              \
      taco::util::TimeResults result = timer.getResult();        \
      cout << NAME << " " << result << " ms" << endl;            \
      TIMER=result;                                              \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

enum class Kind {
      DENSE,
      SPARSE,
      RLE,
      LZ77
};

Func getCopyFunc();
Func getPlusFunc();
Func getPlusRleFunc();

template <class T>
Kernel getKernel(IndexStmt indexStmt, Tensor<T> t){
  std::shared_ptr<ir::Module> module = t.getModule();
  void* compute  = module->getFuncPtr("compute");
  void* assemble = module->getFuncPtr("assemble");
  void* evaluate = module->getFuncPtr("evaluate");
  Kernel k(indexStmt, module, evaluate, assemble, compute);
  return k;
}

std::pair<Tensor<uint8_t>, size_t> read_png(int i, Kind kind);
std::pair<Tensor<uint8_t>, size_t> read_rgb_png(int i, Kind kind);

std::vector<uint8_t> unpackLZ77_bytes(std::vector<uint8_t> bytes, int& numVals, bool unpack = true);

uint32_t saveTensor(std::vector<unsigned char> valsVec, std::string path);


unsigned decode(std::vector<unsigned char>& out, std::vector<unsigned char>& c_out, std::vector<int>& pos, unsigned& w, unsigned& h,
                const std::vector<unsigned char>& in, LodePNGColorType colortype = LCT_GREY, unsigned bitdepth = 8);

template <typename T>
union GetBytes {
    T value;
    uint8_t bytes[sizeof(T)];
};

using Repeat = std::pair<uint16_t, uint16_t>;
template <class T>
using TempValue = std::variant<T,Repeat>;

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T>
T get_value(const std::vector<uint8_t>& bytes, size_t pos){
  T* ptr = (T*) &bytes[pos];
  return *ptr;
}

template <typename T>
void set_value(std::vector<uint8_t>& bytes, size_t pos, T val){
  GetBytes<T> gb{val};
  for (unsigned long i_=0; i_<sizeof(T); i_++){
    bytes[pos+i_] = gb.bytes[i_];
  }
}

template <typename T>
void push_back(T arg, std::vector<uint8_t>& bytes, size_t& curr_count, bool& isValues, bool check = false){
  GetBytes<T> gb;
  gb.value = arg;

  uint16_t mask = (uint16_t)0x7FFF;
  uint16_t count = 0;
  if (check) {
    if (isValues && ((count = get_value<uint16_t>(bytes, curr_count)) < mask)) {
      auto temp_curr_count = curr_count;
      set_value<uint16_t>(bytes, curr_count, count + 1);
      push_back<T>(arg, bytes, curr_count, isValues, false);
      curr_count = temp_curr_count;
    } else {
      push_back<uint16_t>(1, bytes, curr_count, isValues, false);
      auto temp_curr_count = size_t(bytes.empty() ? 0 : bytes.size()-2);
      push_back<T>(arg, bytes, curr_count, isValues, false);
      curr_count = temp_curr_count;
    }
    isValues = true;
  } else {
    for (unsigned long i_=0; i_<sizeof(T); i_++){
      bytes.push_back(gb.bytes[i_]);
    }
    isValues = false;
    curr_count = 0;
  }
}

template <typename T>
std::vector<uint8_t> packLZ77_bytes(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes;
  size_t curr_count = 0;
  bool isValues = false;
  const auto runMask = (uint16_t)~0x7FFF;
  for (auto& val : vals){
    std::visit(overloaded {
            [&](T arg) { push_back(arg, bytes, curr_count, isValues, true); },
            [&](std::pair<uint16_t, uint16_t> arg) {
                push_back<uint16_t>(arg.second | runMask, bytes, curr_count, isValues);
                push_back<uint16_t>(arg.first, bytes, curr_count, isValues);
            }
    }, val);
  }
  return bytes;
}

Index makeLZ77Index(const std::vector<int>& rowptr, int numDense = 0);
Index makeLZ77ImgIndex(const std::vector<int>& rowptr);

template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
  // taco_uassert(dimensions.size() == 1);
  std::vector<ModeFormatPack> fs;
  for (int i = 0; i< dimensions.size()-1; i++) fs.push_back(Dense);
  fs.push_back(LZ77);
  Tensor<T> tensor(name, dimensions, Format(fs));
  auto storage = tensor.getStorage();
  if (dimensions.size()==2) {
    storage.setIndex(makeLZ77ImgIndex(pos));
  } else {
    storage.setIndex(makeLZ77Index(pos));
  }
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}
#endif