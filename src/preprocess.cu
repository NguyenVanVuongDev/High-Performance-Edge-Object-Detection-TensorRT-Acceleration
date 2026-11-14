#include "preprocess.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void fused_letterbox_bgr2rgb_norm_chw_kernel(
    const uint8_t* __restrict__ src,
    int src_w,
    int src_h,
    int src_step,
    float* __restrict__ dst,
    int dst_w,
    int dst_h,
    float scale_x,
    float scale_y,
    int pad_w,
    int pad_h,
    int new_unpad_w,
    int new_unpad_h
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    const int plane_size = dst_w * dst_h;
    const int out_idx = dy * dst_w + dx;

    // Kiểm tra nếu điểm ảnh nằm trong vùng Letterbox padding (màu xám 114/255.0f)
    if (dx < pad_w || dx >= (pad_w + new_unpad_w) ||
        dy < pad_h || dy >= (pad_h + new_unpad_h)) {
        const float pad_val = 114.0f * (1.0f / 255.0f);
        dst[out_idx] = pad_val;
        dst[plane_size + out_idx] = pad_val;
        dst[2 * plane_size + out_idx] = pad_val;
        return;
    }

    // Ánh xạ tọa độ tâm điểm ảnh từ vùng unpad về ảnh gốc (Bilinear Interpolation chuẩn OpenCV)
    float sx = ((float)(dx - pad_w) + 0.5f) * scale_x - 0.5f;
    float sy = ((float)(dy - pad_h) + 0.5f) * scale_y - 0.5f;

    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float wx = sx - (float)x0;
    float wy = sy - (float)y0;
    float w00 = (1.0f - wx) * (1.0f - wy);
    float w01 = wx * (1.0f - wy);
    float w10 = (1.0f - wx) * wy;
    float w11 = wx * wy;

    const uint8_t* p00 = src + y0 * src_step + x0 * 3;
    const uint8_t* p01 = src + y0 * src_step + x1 * 3;
    const uint8_t* p10 = src + y1 * src_step + x0 * 3;
    const uint8_t* p11 = src + y1 * src_step + x1 * 3;

    // Nguồn OpenCV là BGR (0: B, 1: G, 2: R) -> Đích TensorRT là RGB phẳng [R, G, B] chia 255.0f
    const float norm = 1.0f / 255.0f;
    float b = (w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]) * norm;
    float g = (w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]) * norm;
    float r = (w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]) * norm;

    dst[out_idx] = r;
    dst[plane_size + out_idx] = g;
    dst[2 * plane_size + out_idx] = b;
}

void Preprocessor::launchPreprocessKernel(
    const uint8_t* d_src,
    int src_w,
    int src_h,
    int src_step,
    float* d_dst,
    int dst_w,
    int dst_h,
    const LetterboxInfo& info,
    cudaStream_t stream
) {
    dim3 block(32, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    float scale_x = (float)src_w / (float)info.new_unpad_w;
    float scale_y = (float)src_h / (float)info.new_unpad_h;

    fused_letterbox_bgr2rgb_norm_chw_kernel<<<grid, block, 0, stream>>>(
        d_src,
        src_w,
        src_h,
        src_step,
        d_dst,
        dst_w,
        dst_h,
        scale_x,
        scale_y,
        info.pad_w,
        info.pad_h,
        info.new_unpad_w,
        info.new_unpad_h
    );
}
