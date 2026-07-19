cmake_minimum_required(VERSION 3.18)
project(bevformer_activity_profiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS /usr/local/cuda/extras/CUPTI/include $ENV{CUDA_HOME}/extras/CUPTI/include)
find_library(CUPTI_LIBRARY cupti
  HINTS /usr/local/cuda/extras/CUPTI/lib64 $ENV{CUDA_HOME}/extras/CUPTI/lib64)
find_path(CUDA_INCLUDE_DIR cuda.h
  HINTS /usr/local/cuda/include $ENV{CUDA_HOME}/include)
find_library(CUDA_LIBRARY cuda
  HINTS /usr/local/cuda/lib64 $ENV{CUDA_HOME}/lib64)
find_library(CUDART_LIBRARY cudart
  HINTS /usr/local/cuda/lib64 $ENV{CUDA_HOME}/lib64)

if(NOT CUPTI_INCLUDE_DIR OR NOT CUPTI_LIBRARY)
  message(FATAL_ERROR "CUPTI was not found. Set CUDA_HOME or install CUDA Toolkit with CUPTI.")
endif()

add_library(bevformer_activity_profiler SHARED src/activity_profiler.cpp)
target_include_directories(bevformer_activity_profiler PRIVATE ${CUPTI_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(bevformer_activity_profiler PRIVATE ${CUPTI_LIBRARY} ${CUDA_LIBRARY} ${CUDART_LIBRARY} dl pthread)
