#include "preprocess.h"
#include <algorithm>

LetterboxInfo Preprocessor::letterbox(const cv::Mat& src, cv::Mat& dst, int target_w, int target_h) {
    LetterboxInfo info;
    info.in_w = src.cols;
    info.in_h = src.rows;

    float r = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_unpad_w = (int)std::round(src.cols * r);
    int new_unpad_h = (int)std::round(src.rows * r);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    int dw = target_w - new_unpad_w;
    int dh = target_h - new_unpad_h;

    info.scale = r;
    info.pad_w = dw / 2;
    info.pad_h = dh / 2;

    int top = info.pad_h;
    int bottom = dh - top;
    int left = info.pad_w;
    int right = dw - left;

    cv::copyMakeBorder(resized, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return info;
}

void Preprocessor::process(const cv::Mat& src, float* host_buffer, LetterboxInfo& info, int target_w, int target_h) {
    cv::Mat letterboxed;
    info = letterbox(src, letterboxed, target_w, target_h);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    int channel_size = target_w * target_h;
    float* r_plane = host_buffer;
    float* g_plane = host_buffer + channel_size;
    float* b_plane = host_buffer + 2 * channel_size;

    // HWC -> CHW và chia 255.0f
    for (int h = 0; h < target_h; ++h) {
        const uchar* ptr = rgb.ptr<uchar>(h);
        for (int w = 0; w < target_w; ++w) {
            int idx = h * target_w + w;
            r_plane[idx] = (float)ptr[3 * w + 0] / 255.0f;
            g_plane[idx] = (float)ptr[3 * w + 1] / 255.0f;
            b_plane[idx] = (float)ptr[3 * w + 2] / 255.0f;
        }
    }
}
