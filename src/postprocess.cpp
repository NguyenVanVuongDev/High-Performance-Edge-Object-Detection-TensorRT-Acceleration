#include "postprocess.h"
#include <algorithm>
#include <cmath>

std::vector<Detection> Postprocessor::process(
    const float* __restrict__ output_tensor,
    const LetterboxInfo& info,
    float conf_threshold,
    float nms_threshold,
    int num_classes,
    int num_anchors
) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(128);
    confidences.reserve(128);
    class_ids.reserve(128);

    const float* __restrict__ cx_ptr = output_tensor + 0 * num_anchors;
    const float* __restrict__ cy_ptr = output_tensor + 1 * num_anchors;
    const float* __restrict__ w_ptr  = output_tensor + 2 * num_anchors;
    const float* __restrict__ h_ptr  = output_tensor + 3 * num_anchors;
    const float* __restrict__ cls_base = output_tensor + 4 * num_anchors;

    const float inv_scale = 1.0f / info.scale;
    const float max_w = (float)info.in_w;
    const float max_h = (float)info.in_h;

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = conf_threshold;
        int best_class = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = cls_base[c * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (best_class != -1) {
            float cx = cx_ptr[i];
            float cy = cy_ptr[i];
            float w  = w_ptr[i];
            float h  = h_ptr[i];

            float x1 = (cx - w * 0.5f - (float)info.pad_w) * inv_scale;
            float y1 = (cy - h * 0.5f - (float)info.pad_h) * inv_scale;
            float bw = w * inv_scale;
            float bh = h * inv_scale;

            x1 = std::max(0.0f, std::min(x1, max_w));
            y1 = std::max(0.0f, std::min(y1, max_h));
            bw = std::min(bw, max_w - x1);
            bh = std::min(bh, max_h - y1);

            boxes.emplace_back((int)std::round(x1), (int)std::round(y1), (int)std::round(bw), (int)std::round(bh));
            confidences.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx], class_ids[idx]});
    }
    return detections;
}
