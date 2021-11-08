cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(opencv-bench)
set(CMAKE_BUILD_TYPE "release")
option(LANKA "Build benchmarks for the Lanka cluster." OFF)

include_directories(${CMAKE_CURRENT_LIST_DIR}/opencv/install/include/opencv4/)

file(GLOB TEST_SOURCES *.cpp *.c *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++17")

add_executable(opencv-bench ${TEST_SOURCES})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/opencv_bench;${CMAKE_MODULE_PATH}")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

file(GLOB LIBS_FILES ${CMAKE_CURRENT_LIST_DIR}/opencv/install/lib/*.so)


target_link_libraries(opencv-bench ${OpenCV_LIBS} ${LIBS_FILES})
# Only link stdc++fs on lanka. Mac's don't need this set.
if (NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(opencv-bench stdc++fs)
endif ()

