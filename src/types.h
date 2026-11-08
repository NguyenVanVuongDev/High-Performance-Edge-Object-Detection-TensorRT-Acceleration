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
};
