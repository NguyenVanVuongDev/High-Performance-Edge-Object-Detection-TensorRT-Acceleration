cmake_minimum_required(VERSION 3.18)
project(TensorRTYoloEdge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-deprecated-declarations")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /usr/include/x86_64-linux-gnu
)

link_directories(
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)

add_executable(yolo_infer
    src/main.cpp
    src/engine.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)

target_link_libraries(yolo_infer
    ${OpenCV_LIBS}
    nvinfer
    cudart
    dl
)
