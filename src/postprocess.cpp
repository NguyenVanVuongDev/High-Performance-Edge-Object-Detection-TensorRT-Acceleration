#include "postprocess.h"
#include <algorithm>

std::vector<Detection> Postprocessor::process(
    const float* output_tensor,
    const LetterboxInfo& info,
    float conf_threshold,
    float nms_threshold,
    int num_classes,
    int num_anchors
) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // Output tensor có kích thước: [84, 8400]
    // 4 hàng đầu: cx, cy, w, h
    // 80 hàng tiếp theo: class probabilities
    for (int i = 0; i < num_anchors; ++i) {
        float max_score = -1.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = output_tensor[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score >= conf_threshold) {
            float cx = output_tensor[0 * num_anchors + i];
            float cy = output_tensor[1 * num_anchors + i];
            float w  = output_tensor[2 * num_anchors + i];
            float h  = output_tensor[3 * num_anchors + i];

            // Chuyển cx, cy, w, h -> x1, y1, w, h
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;

            // Map ngược về kích thước ảnh gốc
            x1 = (x1 - info.pad_w) / info.scale;
            y1 = (y1 - info.pad_h) / info.scale;
            w  = w / info.scale;
            h  = h / info.scale;

            // Clamp tọa độ
            x1 = std::max(0.0f, std::min(x1, (float)info.in_w));
            y1 = std::max(0.0f, std::min(y1, (float)info.in_h));
            w  = std::min(w, (float)info.in_w - x1);
            h  = std::min(h, (float)info.in_h - y1);

            boxes.emplace_back(cv::Rect((int)x1, (int)y1, (int)w, (int)h));
            confidences.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    return detections;
}
