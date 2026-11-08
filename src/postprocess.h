#pragma once
#include "types.h"
#include <vector>

class Postprocessor {
public:
    static std::vector<Detection> process(
        const float* output_tensor,
        const LetterboxInfo& info,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f,
        int num_classes = 80,
        int num_anchors = 8400
    );
};
