#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

class Preprocessor {
public:
    static LetterboxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_w = 640, int target_h = 640);
    static void process(const cv::Mat& src, float* host_buffer, LetterboxInfo& info, int target_w = 640, int target_h = 640);
};
