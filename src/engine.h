#pragma once
#include "types.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TRT] %s\n", msg);
        }
    }
};

class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    bool load(const std::string& engine_path);

    // Chế độ 1: Fused CUDA Preprocess + EnqueueV3 trên cùng 1 CUDA Stream (Tối ưu hóa Ngày 3)
    void inferFusedGPU(const cv::Mat& src, LetterboxInfo& info, float& out_pre_ms, float& out_infer_ms);

    // Chế độ 2: CPU Preprocess + H2D float32 + EnqueueV3 (Baseline Ngày 2 để so sánh A/B)
    void inferAsyncCPUInput();
    void synchronize();

    float* getHostInput() { return host_input_; }
    float* getHostOutput() { return host_output_; }

private:
    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_after_pre_ = nullptr;
    cudaEvent_t ev_after_infer_ = nullptr;

    float* host_input_ = nullptr;
    float* host_output_ = nullptr;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;

    // Bộ nhớ đệm Pinned & Device cho ảnh thô uint8_t (tối đa 4K UHD: 3840x2160x3 bytes)
    uint8_t* host_raw_img_ = nullptr;
    uint8_t* device_raw_img_ = nullptr;
    size_t max_raw_bytes_ = 3840 * 2160 * 3;

    size_t input_size_ = 1 * 3 * 640 * 640 * sizeof(float);
    size_t output_size_ = 1 * 84 * 8400 * sizeof(float);

    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;
};
