cmake_minimum_required(VERSION 3.18)
project(TensorRTYolo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Tối ưu kiến trúc GPU Tesla T4 (Turing sm_75) & tắt cảnh báo deprecated từ header TensorRT
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG -Xcompiler=-Wno-deprecated-declarations")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/x86_64-linux-gnu
    /usr/local/include
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/cuda/lib64
)

add_executable(yolo_infer
    src/main.cpp
    src/engine.cpp
    src/preprocess.cpp
    src/preprocess.cu
    src/postprocess.cpp
)

set_target_properties(yolo_infer PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(yolo_infer
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    CUDA::cudart
    dl
)
