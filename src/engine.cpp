#include "engine.h"
#include <fstream>
#include <iostream>

TRTEngine::TRTEngine() {}

TRTEngine::~TRTEngine() {
    if (stream_) cudaStreamDestroy(stream_);
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

    cudaStreamCreate(&stream_);

    // Cấp phát Host Pinned Memory (Zero-copy / Fast DMA transfer)
    cudaMallocHost((void**)&host_input_, input_size_);
    cudaMallocHost((void**)&host_output_, output_size_);

    // Cấp phát Device VRAM
    cudaMalloc(&device_input_, input_size_);
    cudaMalloc(&device_output_, output_size_);

    // Lấy tên I/O tensors (Hỗ trợ TensorRT 10.x & 11.x)
    int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
        } else {
            output_name_ = name;
        }
    }

    // Set địa chỉ tensor cho context (API chuẩn enqueueV3)
    context_->setInputTensorAddress(input_name_, device_input_);
    context_->setOutputTensorAddress(output_name_, device_output_);

    std::cout << "[Engine] Successfully loaded: " << engine_path << std::endl;
    return true;
}

void TRTEngine::inferAsync() {
    // 1. Host -> Device Copy bất đồng bộ
    cudaMemcpyAsync(device_input_, host_input_, input_size_, cudaMemcpyHostToDevice, stream_);

    // 2. Enqueue Kernel thực thi trên GPU
    context_->enqueueV3(stream_);

    // 3. Device -> Host Copy kết quả
    cudaMemcpyAsync(host_output_, device_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
}

void TRTEngine::synchronize() {
    cudaStreamSynchronize(stream_);
}
