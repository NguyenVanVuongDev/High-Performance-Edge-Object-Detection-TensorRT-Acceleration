#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cstdint>

class Preprocessor {
public:
    static LetterboxInfo computeLetterboxInfo(int src_w, int src_h, int target_w = 640, int target_h = 640);

    // Baseline CPU Pre-processing (OpenCV resize + copyMakeBorder + cvtColor + CPU CHW loop)
    static void processCPU(
        const cv::Mat& src,
        float* host_buffer,
        LetterboxInfo& info,
        int target_w = 640,
        int target_h = 640
    );

    // Optimized GPU Pre-processing (Fused CUDA Kernel: Bilinear Letterbox + BGR2RGB + Norm + HWC2CHW)
    static void launchPreprocessKernel(
        const uint8_t* d_src,
        int src_w,
        int src_h,
        int src_step,
        float* d_dst,
        int dst_w,
        int dst_h,
        const LetterboxInfo& info,
        cudaStream_t stream
    );
};
