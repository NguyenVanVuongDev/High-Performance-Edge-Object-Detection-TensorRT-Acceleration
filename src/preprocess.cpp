#include "preprocess.h"
#include <algorithm>
#include <cmath>

LetterboxInfo Preprocessor::computeLetterboxInfo(int src_w, int src_h, int target_w, int target_h) {
    LetterboxInfo info;
    info.in_w = src_w;
    info.in_h = src_h;

    float r = std::min((float)target_w / src_w, (float)target_h / src_h);
    info.new_unpad_w = (int)std::round(src_w * r);
    info.new_unpad_h = (int)std::round(src_h * r);

    int dw = target_w - info.new_unpad_w;
    int dh = target_h - info.new_unpad_h;

    info.scale = r;
    info.pad_w = dw / 2;
    info.pad_h = dh / 2;
    return info;
}

void Preprocessor::processCPU(const cv::Mat& src, float* host_buffer, LetterboxInfo& info, int target_w, int target_h) {
    info = computeLetterboxInfo(src.cols, src.rows, target_w, target_h);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(info.new_unpad_w, info.new_unpad_h), 0, 0, cv::INTER_LINEAR);

    int dw = target_w - info.new_unpad_w;
    int dh = target_h - info.new_unpad_h;
    int top = info.pad_h;
    int bottom = dh - top;
    int left = info.pad_w;
    int right = dw - left;

    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    int channel_size = target_w * target_h;
    float* __restrict__ r_plane = host_buffer;
    float* __restrict__ g_plane = host_buffer + channel_size;
    float* __restrict__ b_plane = host_buffer + 2 * channel_size;

    const float norm = 1.0f / 255.0f;
    for (int h = 0; h < target_h; ++h) {
        const uchar* __restrict__ ptr = rgb.ptr<uchar>(h);
        for (int w = 0; w < target_w; ++w) {
            int idx = h * target_w + w;
            r_plane[idx] = (float)ptr[3 * w + 0] * norm;
            g_plane[idx] = (float)ptr[3 * w + 1] * norm;
            b_plane[idx] = (float)ptr[3 * w + 2] * norm;
        }
    }
}
