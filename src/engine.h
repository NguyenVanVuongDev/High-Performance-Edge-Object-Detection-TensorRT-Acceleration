#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>
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
    void inferAsync();
    void synchronize();

    float* getHostInput() { return host_input_; }
    float* getHostOutput() { return host_output_; }

    size_t getInputSize() const { return input_size_; }
    size_t getOutputSize() const { return output_size_; }

private:
    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    cudaStream_t stream_ = nullptr;

    float* host_input_ = nullptr;
    float* host_output_ = nullptr;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;

    size_t input_size_ = 1 * 3 * 640 * 640 * sizeof(float);
    size_t output_size_ = 1 * 84 * 8400 * sizeof(float);

    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;
};
