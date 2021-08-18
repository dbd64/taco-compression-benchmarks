#include "bench.h"

#include <cstdlib>
#include <iostream>

#include "taco/tensor.h"
#include "taco/util/strings.h"

std::string getEnvVar(std::string varname) {
  auto path = std::getenv(varname.c_str());
  if (path == nullptr) {
    return "";
  }
  return std::string(path);
}

std::string getTacoTensorPath() {
  std::string result = getEnvVar("TACO_TENSOR_PATH");
  if (result == "") {
    assert(false && "TACO_TENSOR_PATH is unset");
  }
  return cleanPath(result);
}

std::string getOutputPath() {
  auto result = getEnvVar("OUTPUT_PATH");
  if (result != "") {
    result = cleanPath(result);
  }
  return result;
}

std::string getValidationOutputPath() {
  auto result = getEnvVar("VALIDATION_OUTPUT_PATH");
  if (result != "") {
    result = cleanPath(result);
  }
  return result;
}

std::string cleanPath(std::string path) {
  std::string result(path);
  if (result[result.size() - 1] != '/') {
    result += "/";
  }
  return result;
}