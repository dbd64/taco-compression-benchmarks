cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
option(LANKA "Build benchmarks for the Lanka cluster." OFF)

add_subdirectory(taco)
add_subdirectory(benchmark)

include_directories(taco taco/include benchmark/include taco/src)

file(GLOB TEST_SOURCES *.cpp *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -ggdb -std=c++17")

add_executable(taco-bench ${TEST_SOURCES})
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
# Only link stdc++fs on lanka. Mac's don't need this set.
if (LANKA)
  target_link_libraries(taco-bench stdc++fs)
endif(LANKA)
