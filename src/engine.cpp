#include "engine.h"
#include "preprocess.h"
#include <fstream>
#include <iostream>
#include <cstring>

TRTEngine::TRTEngine() {}

TRTEngine::~TRTEngine() {
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_after_pre_) cudaEventDestroy(ev_after_pre_);
    if (ev_after_infer_) cudaEventDestroy(ev_after_infer_);
    if (stream_) cudaStreamDestroy(stream_);
    if (device_raw_img_) cudaFree(device_raw_img_);
    if (host_raw_img_) cudaFreeHost(host_raw_img_);
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    if (host_input_) cudaFreeHost(host_input_);
    if (host_output_) cudaFreeHost(host_output_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

bool TRTEngine::load(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[Engine] Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) return false;

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) return false;

    context_ = engine_->createExecutionContext();
    if (!context_) return false;

    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_after_pre_);
    cudaEventCreate(&ev_after_infer_);

    cudaMallocHost((void**)&host_input_, input_size_);
    cudaMallocHost((void**)&host_output_, output_size_);
    cudaMalloc(&device_input_, input_size_);
    cudaMalloc(&device_output_, output_size_);

    cudaMallocHost((void**)&host_raw_img_, max_raw_bytes_);
    cudaMalloc((void**)&device_raw_img_, max_raw_bytes_);

    int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
        } else {
            output_name_ = name;
        }
    }

    context_->setInputTensorAddress(input_name_, device_input_);
    context_->setOutputTensorAddress(output_name_, device_output_);

    std::cout << "[Engine] Loaded: " << engine_path
              << " | IO Tensors: " << input_name_ << " -> " << output_name_ << std::endl;
    return true;
}

void TRTEngine::inferFusedGPU(const cv::Mat& src, LetterboxInfo& info, float& out_pre_ms, float& out_infer_ms) {
    info = Preprocessor::computeLetterboxInfo(src.cols, src.rows, 640, 640);

    size_t raw_bytes = src.step[0] * src.rows;
    if (raw_bytes > max_raw_bytes_) {
        cudaFreeHost(host_raw_img_);
        cudaFree(device_raw_img_);
        max_raw_bytes_ = raw_bytes;
        cudaMallocHost((void**)&host_raw_img_, max_raw_bytes_);
        cudaMalloc((void**)&device_raw_img_, max_raw_bytes_);
    }

    // Copy nhanh vào Pinned Memory để DMA H2D đạt tốc độ tối đa
    std::memcpy(host_raw_img_, src.data, raw_bytes);

    cudaEventRecord(ev_start_, stream_);

    // 1. Copy ảnh uint8_t thô từ Host Pinned -> Device VRAM
    cudaMemcpyAsync(device_raw_img_, host_raw_img_, raw_bytes, cudaMemcpyHostToDevice, stream_);

    // 2. Chạy Fused CUDA Kernel Tiền xử lý ngay trên GPU
    Preprocessor::launchPreprocessKernel(
        device_raw_img_,
        src.cols,
        src.rows,
        (int)src.step[0],
        (float*)device_input_,
        640,
        640,
        info,
        stream_
    );
    cudaEventRecord(ev_after_pre_, stream_);

    // 3. Thực thi TensorRT Engine trực tiếp trên device_input_ đã nằm sẵn trong VRAM
    context_->enqueueV3(stream_);

    // 4. Copy kết quả output0 [1, 84, 8400] về Pinned Host Memory
    cudaMemcpyAsync(host_output_, device_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
    cudaEventRecord(ev_after_infer_, stream_);

    // Đồng bộ 1 lần duy nhất ở cuối chuỗi GPU
    cudaStreamSynchronize(stream_);

    cudaEventElapsedTime(&out_pre_ms, ev_start_, ev_after_pre_);
    cudaEventElapsedTime(&out_infer_ms, ev_after_pre_, ev_after_infer_);
}

void TRTEngine::inferAsyncCPUInput() {
    cudaMemcpyAsync(device_input_, host_input_, input_size_, cudaMemcpyHostToDevice, stream_);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(host_output_, device_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
}

void TRTEngine::synchronize() {
    cudaStreamSynchronize(stream_);
}
