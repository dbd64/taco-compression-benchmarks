#include "../bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"
#include "taco/util/timers.h"
#include "../utils.h"
#include "../rapidcsv.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"
#include "../png_reader.h"

#include <random>

using namespace taco;

int main(){
  auto index_str = getEnvVar("INDEX");
  if (index_str == "") {
    std::cout << "No index" << std::endl;
    return;
  }
  int index = std::stoi(index_str);


  auto folder1 = getEnvVar("FOLDER");
  if (folder1 == "") {
    std::cout << "No folder1" << std::endl;
    return;
  }

  int w = 0; int h = 0; int f1_num_vals = 0;

  std::ostringstream stringStream;
  stringStream << folder1;
  if (index < 10){
      stringStream << "00";
  } else if (index < 100){
      stringStream << "0";
  }
  stringStream << index << ".png";
  std::string filename = stringStream.str();

  auto image = raw_image_ma(filename, w, h);

  auto outfile = getEnvVar("OUT_FILE");
  std::ofstream file(outfile, std::ios::out|std::ios::binary);
  std::copy(image.cbegin(), image.cend(),
     ostreambuf_iterator<char>(file));
  return 0;
}
