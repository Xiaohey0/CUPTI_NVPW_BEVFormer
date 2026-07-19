cmake_minimum_required(VERSION 3.18)
project(bevformer_msda_nvpw_replay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
  set(CUDA_HOME_PATH "$ENV{CUDA_HOME}")
else()
  set(CUDA_HOME_PATH /usr/local/cuda)
endif()

set(CUPTI_SAMPLES_DIR ${CUDA_HOME_PATH}/extras/CUPTI/samples)

find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS ${CUDA_HOME_PATH}/extras/CUPTI/include ${CUDA_HOME_PATH}/include)
find_library(CUPTI_LIBRARY cupti
  HINTS ${CUDA_HOME_PATH}/extras/CUPTI/lib64)
find_library(NVPERF_HOST_LIBRARY nvperf_host
  HINTS ${CUDA_HOME_PATH}/extras/CUPTI/lib64)
find_library(NVPERF_TARGET_LIBRARY nvperf_target
  HINTS ${CUDA_HOME_PATH}/extras/CUPTI/lib64)
find_library(CUDA_LIBRARY cuda
  HINTS ${CUDA_HOME_PATH}/lib64 ${CUDA_HOME_PATH}/lib64/stubs)
find_library(CUDART_LIBRARY cudart
  HINTS ${CUDA_HOME_PATH}/lib64)

if(NOT CUPTI_INCLUDE_DIR OR NOT CUPTI_LIBRARY)
  message(FATAL_ERROR "CUPTI was not found. Set CUDA_HOME to a CUDA Toolkit with CUPTI.")
endif()

add_library(profilerHostUtil STATIC
  ${CUPTI_SAMPLES_DIR}/extensions/src/profilerhost_util/Eval.cpp
  ${CUPTI_SAMPLES_DIR}/extensions/src/profilerhost_util/Metric.cpp
  ${CUPTI_SAMPLES_DIR}/extensions/src/profilerhost_util/List.cpp
)
target_include_directories(profilerHostUtil PUBLIC
  ${CUPTI_INCLUDE_DIR}
  ${CUDA_HOME_PATH}/include
  ${CUPTI_SAMPLES_DIR}/extensions/include/profilerhost_util
  ${CUPTI_SAMPLES_DIR}/extensions/include/c_util
  ${CUPTI_SAMPLES_DIR}/common
)
target_link_libraries(profilerHostUtil PUBLIC ${NVPERF_HOST_LIBRARY} ${NVPERF_TARGET_LIBRARY})

add_library(bevformer_msda_nvpw_replay SHARED src/msda_nvpw_replay.cpp)
target_include_directories(bevformer_msda_nvpw_replay PRIVATE
  ${CUPTI_INCLUDE_DIR}
  ${CUDA_HOME_PATH}/include
  ${CUPTI_SAMPLES_DIR}/extensions/include/profilerhost_util
  ${CUPTI_SAMPLES_DIR}/extensions/include/c_util
  ${CUPTI_SAMPLES_DIR}/common
)
target_link_libraries(bevformer_msda_nvpw_replay PRIVATE
  profilerHostUtil
  ${CUPTI_LIBRARY}
  ${CUDA_LIBRARY}
  ${CUDART_LIBRARY}
  ${NVPERF_HOST_LIBRARY}
  ${NVPERF_TARGET_LIBRARY}
  pthread
)
