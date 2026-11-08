#include "engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>
#include <iostream>
#include <iomanip>

const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./yolo_infer <engine_path> <image_path> <output_path> [iterations]" << std::endl;
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path  = argv[2];
    std::string output_path = argv[3];
    int iterations = (argc >= 5) ? std::stoi(argv[4]) : 10;

    TRTEngine engine;
    if (!engine.load(engine_path)) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // Warm-up 10 lần
    LetterboxInfo info;
    for (int i = 0; i < 10; ++i) {
        Preprocessor::process(image, engine.getHostInput(), info);
        engine.inferAsync();
        engine.synchronize();
    }

    // Benchmark đo đạc
    double total_pre = 0, total_infer = 0, total_post = 0;
    std::vector<Detection> detections;

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Preprocessor::process(image, engine.getHostInput(), info);
        auto t1 = std::chrono::high_resolution_clock::now();

        engine.inferAsync();
        engine.synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        detections = Postprocessor::process(engine.getHostOutput(), info, 0.25f, 0.45f);
        auto t3 = std::chrono::high_resolution_clock::now();

        total_pre   += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_infer += std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_post  += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    double avg_pre   = total_pre / iterations;
    double avg_infer = total_infer / iterations;
    double avg_post  = total_post / iterations;
    double avg_e2e   = avg_pre + avg_infer + avg_post;
    double fps       = 1000.0 / avg_e2e;

    std::cout << "\n================ NATIVE C++ / TENSORRT BENCHMARK ================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Engine:                 " << engine_path << std::endl;
    std::cout << "Input Resolution:       " << image.cols << "x" << image.rows << " -> 640x640" << std::endl;
    std::cout << "Detected Objects:       " << detections.size() << std::endl;
    for (size_t d = 0; d < detections.size(); ++d) {
        const auto& det = detections[d];
        std::string label = (det.class_id < (int)COCO_CLASSES.size()) ? COCO_CLASSES[det.class_id] : std::to_string(det.class_id);
        std::cout << "  - [" << (d + 1) << "] " << label << " (" << cv::format("%.2f", det.confidence)
                  << ") | Box: [" << det.box.x << ", " << det.box.y << ", "
                  << det.box.width << ", " << det.box.height << "]" << std::endl;
    }
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Pre-process Latency:    " << avg_pre << " ms" << std::endl;
    std::cout << "Inference Latency:      " << avg_infer << " ms (GPU Kernel)" << std::endl;
    std::cout << "Post-process (NMS):     " << avg_post << " ms" << std::endl;
    std::cout << "End-to-End Latency:     " << avg_e2e << " ms" << std::endl;
    std::cout << "Throughput (FPS):       " << fps << " frames/s" << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    // Vẽ Bounding Box và Label
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = (det.class_id < (int)COCO_CLASSES.size()) ? COCO_CLASSES[det.class_id] : std::to_string(det.class_id);
        label += " " + cv::format("%.2f", det.confidence);

        int base_line;
        cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::rectangle(image, cv::Rect(det.box.x, det.box.y - size.height - 4, size.width, size.height + 4), cv::Scalar(0, 255, 0), -1);
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite(output_path, image);
    std::cout << "[Output] Result image saved to: " << output_path << std::endl;
    return 0;
}

