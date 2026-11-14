#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

struct LetterboxInfo {
    float scale;
    int pad_w;
    int pad_h;
    int in_w;
    int in_h;
    int new_unpad_w;
    int new_unpad_h;
};

struct FrameMetrics {
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double post_ms = 0.0;
    double e2e_ms = 0.0;
};
