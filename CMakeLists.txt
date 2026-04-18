cmake_minimum_required(VERSION 3.15)
project(perceptual_occlusion_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

# ONNX Runtime — supports both Apple Silicon and Intel Homebrew paths
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS
        /opt/homebrew/include/onnxruntime
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime)

find_library(ORT_LIB onnxruntime
    PATHS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "Install with: brew install onnxruntime")
endif()

# Metal frameworks (macOS only)
find_library(METAL_LIB      Metal      REQUIRED)
find_library(FOUNDATION_LIB Foundation REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIR} src)

add_executable(engine
    src/main.cpp
    src/core/Engine.cpp
    src/camera/Camera.cpp
    src/perception/DepthEstimator.cpp
    src/perception/DepthPostProcessor.cpp
    src/rendering/MetalContext.mm
    src/rendering/Renderer.mm
    src/compositing/OcclusionPass.cpp
)

# Exposes project root so Engine.cpp can locate assets/
target_compile_definitions(engine PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

target_link_libraries(engine
    ${OpenCV_LIBS}
    ${ORT_LIB}
    ${METAL_LIB}
    ${FOUNDATION_LIB}
)
