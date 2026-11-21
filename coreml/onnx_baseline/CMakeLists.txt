cmake_minimum_required(VERSION 3.20)
project(onnx_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Same ONNX Runtime lookup as the main project
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /opt/homebrew/include/onnxruntime /usr/local/include/onnxruntime /usr/include/onnxruntime)
find_library(ORT_LIB onnxruntime PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Install with: brew install onnxruntime")
endif()

add_executable(onnx_bench onnx_bench.cpp)
target_include_directories(onnx_bench PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIR})
target_link_libraries(onnx_bench PRIVATE ${OpenCV_LIBS} ${ORT_LIB})
