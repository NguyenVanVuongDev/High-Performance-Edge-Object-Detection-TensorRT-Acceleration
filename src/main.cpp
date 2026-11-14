#include "engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

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

static cv::Scalar getClassColor(int class_id) {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 220, 110),   // person - neon green
        cv::Scalar(255, 180, 0),   // bicycle - cyan-blue
        cv::Scalar(255, 110, 40),  // car - royal blue-orange
        cv::Scalar(0, 165, 255),   // motorcycle - orange
        cv::Scalar(210, 80, 255),  // bus/truck - magenta
        cv::Scalar(50, 235, 255)   // yellow
    };
    return palette[class_id % palette.size()];
}

static void drawDetectionsAndHUD(
    cv::Mat& frame,
    const std::vector<Detection>& detections,
    double ema_fps,
    double e2e_ms,
    double pre_ms,
    double infer_ms,
    double post_ms,
    const std::string& mode_tag,
    int frame_idx = -1,
    int total_frames = -1
) {
    // 1. Vẽ Bounding Boxes
    for (const auto& det : detections) {
        cv::Scalar color = getClassColor(det.class_id);
        cv::rectangle(frame, det.box, color, 2, cv::LINE_AA);

        std::string label = (det.class_id < (int)COCO_CLASSES.size()) ? COCO_CLASSES[det.class_id] : std::to_string(det.class_id);
        label += " " + cv::format("%.2f", det.confidence);

        int base_line = 0;
        cv::Size tsize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.52, 1, &base_line);
        int top_y = std::max(det.box.y, tsize.height + 6);
        cv::rectangle(
            frame,
            cv::Point(det.box.x, top_y - tsize.height - 6),
            cv::Point(det.box.x + tsize.width + 8, top_y),
            color,
            -1
        );
        cv::putText(
            frame,
            label,
            cv::Point(det.box.x + 4, top_y - 4),
            cv::FONT_HERSHEY_DUPLEX,
            0.52,
            cv::Scalar(15, 15, 15),
            1,
            cv::LINE_AA
        );
    }

    // 2. Vẽ Bảng HUD bán trong suốt ở góc trên bên trái
    int hud_w = 450;
    int hud_h = (total_frames > 0) ? 172 : 148;
    hud_w = std::min(hud_w, frame.cols - 20);
    hud_h = std::min(hud_h, frame.rows - 20);

    cv::Rect hud_rect(12, 12, hud_w, hud_h);
    cv::Mat roi = frame(hud_rect);
    cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(18, 20, 28));
    cv::addWeighted(overlay, 0.78, roi, 0.22, 0, roi);
    cv::rectangle(frame, hud_rect, cv::Scalar(0, 230, 125), 1, cv::LINE_AA);

    cv::putText(frame, "YOLOv8s | " + mode_tag, cv::Point(24, 38),
                cv::FONT_HERSHEY_DUPLEX, 0.58, cv::Scalar(0, 240, 140), 1, cv::LINE_AA);

    std::string fps_str = cv::format("FPS: %.1f  |  E2E Latency: %.2f ms", ema_fps, e2e_ms);
    cv::putText(frame, fps_str, cv::Point(24, 68),
                cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    std::string stage_str = cv::format("Pre: %.2f ms | TRT GPU: %.2f ms | NMS: %.2f ms", pre_ms, infer_ms, post_ms);
    cv::putText(frame, stage_str, cv::Point(24, 96),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(195, 215, 235), 1, cv::LINE_AA);

    std::string obj_str = cv::format("Objects Detected: %zu  |  Device: NVIDIA Tesla T4", detections.size());
    cv::putText(frame, obj_str, cv::Point(24, 122),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(120, 220, 255), 1, cv::LINE_AA);

    if (total_frames > 0 && frame_idx >= 0) {
        int bar_x = 24, bar_y = 140, bar_w = hud_w - 110, bar_h = 10;
        cv::rectangle(frame, cv::Rect(bar_x, bar_y, bar_w, bar_h), cv::Scalar(70, 75, 90), -1);
        int fill_w = (int)std::round((double)(frame_idx + 1) / total_frames * bar_w);
        cv::rectangle(frame, cv::Rect(bar_x, bar_y, std::min(bar_w, fill_w), bar_h), cv::Scalar(0, 230, 125), -1);
        std::string f_txt = cv::format("%d/%d", frame_idx + 1, total_frames);
        cv::putText(frame, f_txt, cv::Point(bar_x + bar_w + 10, bar_y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    }
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static void saveMetricsJson(
    const std::string& json_path,
    const std::string& mode_name,
    const std::string& engine_path,
    int iterations,
    double avg_pre,
    double avg_infer,
    double avg_post,
    double avg_e2e,
    double p50_infer,
    double p95_infer,
    double p50_e2e,
    double p95_e2e,
    double p99_e2e,
    double fps,
    size_t num_detections
) {
    if (json_path.empty()) return;
    std::ofstream out(json_path);
    if (!out.good()) return;
    out << std::fixed << std::setprecision(4)
        << "{\n"
        << "  \"mode\": \"" << mode_name << "\",\n"
        << "  \"engine\": \"" << engine_path << "\",\n"
        << "  \"iterations\": " << iterations << ",\n"
        << "  \"pre_ms\": " << avg_pre << ",\n"
        << "  \"infer_ms\": " << avg_infer << ",\n"
        << "  \"post_ms\": " << avg_post << ",\n"
        << "  \"e2e_ms\": " << avg_e2e << ",\n"
        << "  \"p50_infer_ms\": " << p50_infer << ",\n"
        << "  \"p95_infer_ms\": " << p95_infer << ",\n"
        << "  \"p50_e2e_ms\": " << p50_e2e << ",\n"
        << "  \"p95_e2e_ms\": " << p95_e2e << ",\n"
        << "  \"p99_e2e_ms\": " << p99_e2e << ",\n"
        << "  \"fps\": " << fps << ",\n"
        << "  \"num_detections\": " << num_detections << "\n"
        << "}\n";
    out.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage:\n"
                  << "  1. Image (CUDA Pre): ./yolo_infer <engine> <img_in> <img_out> [iters] [json_out]\n"
                  << "  2. Image (CPU Pre):  ./yolo_infer --cpu-pre <engine> <img_in> <img_out> [iters] [json_out]\n"
                  << "  3. Video HUD Demo:   ./yolo_infer --video <engine> <vid_in> <vid_out> [json_out]\n";
        return -1;
    }

    std::string first_arg = argv[1];

    // ================= CHẾ ĐỘ VIDEO HUD DEMO (FUSED CUDA PREPROCESS) =================
    if (first_arg == "--video") {
        if (argc < 5) {
            std::cerr << "Usage: ./yolo_infer --video <engine_path> <input_video> <output_video> [json_out]" << std::endl;
            return -1;
        }
        std::string engine_path = argv[2];
        std::string video_in = argv[3];
        std::string video_out = argv[4];
        std::string json_out = (argc >= 6) ? argv[5] : "";

        TRTEngine engine;
        if (!engine.load(engine_path)) return -1;

        cv::VideoCapture cap(video_in);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open input video: " << video_in << std::endl;
            return -1;
        }

        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double src_fps = cap.get(cv::CAP_PROP_FPS);
        if (src_fps <= 0.0) src_fps = 30.0;
        int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

        cv::VideoWriter writer(
            video_out,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            src_fps,
            cv::Size(width, height)
        );

        cv::Mat frame;
        LetterboxInfo info;
        // Warm-up 15 vòng trên frame đầu
        if (cap.read(frame) && !frame.empty()) {
            float dummy_pre = 0, dummy_inf = 0;
            for (int w = 0; w < 15; ++w) {
                engine.inferFusedGPU(frame, info, dummy_pre, dummy_inf);
            }
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        std::vector<double> pre_vec, infer_vec, post_vec, e2e_vec;
        double ema_fps = 0.0;
        int frame_idx = 0;
        size_t last_det_count = 0;

        while (cap.read(frame) && !frame.empty()) {
            float gpu_pre_ms = 0.0f, gpu_infer_ms = 0.0f;
            auto t0 = std::chrono::high_resolution_clock::now();

            engine.inferFusedGPU(frame, info, gpu_pre_ms, gpu_infer_ms);

            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = Postprocessor::process(engine.getHostOutput(), info, 0.25f, 0.45f);
            auto t2 = std::chrono::high_resolution_clock::now();

            double post_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double e2e_ms  = std::chrono::duration<double, std::milli>(t2 - t0).count();
            double inst_fps = 1000.0 / std::max(0.01, e2e_ms);
            ema_fps = (frame_idx == 0) ? inst_fps : (0.85 * ema_fps + 0.15 * inst_fps);

            pre_vec.push_back(gpu_pre_ms);
            infer_vec.push_back(gpu_infer_ms);
            post_vec.push_back(post_ms);
            e2e_vec.push_back(e2e_ms);
            last_det_count = detections.size();

            drawDetectionsAndHUD(
                frame, detections, ema_fps, e2e_ms,
                gpu_pre_ms, gpu_infer_ms, post_ms,
                "C++ TensorRT FP16 + CUDA Pre", frame_idx, total_frames
            );
            writer.write(frame);
            frame_idx++;
        }

        cap.release();
        writer.release();

        auto mean_of = [](const std::vector<double>& v) {
            return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };
        double avg_pre = mean_of(pre_vec);
        double avg_inf = mean_of(infer_vec);
        double avg_post = mean_of(post_vec);
        double avg_e2e = mean_of(e2e_vec);
        double avg_fps = 1000.0 / std::max(0.001, avg_e2e);

        std::cout << "\n================ NATIVE C++ LIVE VIDEO HUD BENCHMARK ================" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Processed Frames:       " << frame_idx << " (" << width << "x" << height << ")" << std::endl;
        std::cout << "Avg CUDA Pre-process:   " << avg_pre << " ms" << std::endl;
        std::cout << "Avg TRT GPU Inference:  " << avg_inf << " ms (p50=" << percentile(infer_vec, 50)
                  << " ms, p95=" << percentile(infer_vec, 95) << " ms)" << std::endl;
        std::cout << "Avg Post-process (NMS): " << avg_post << " ms" << std::endl;
        std::cout << "Avg E2E Latency:        " << avg_e2e << " ms (p50=" << percentile(e2e_vec, 50)
                  << " ms, p95=" << percentile(e2e_vec, 95) << " ms, p99=" << percentile(e2e_vec, 99) << " ms)" << std::endl;
        std::cout << "Throughput (FPS):       " << avg_fps << " frames/s" << std::endl;
        std::cout << "Saved HUD Video:        " << video_out << std::endl;
        std::cout << "====================================================================\n" << std::endl;

        saveMetricsJson(
            json_out, "Video_720p_CUDA_Pre", engine_path, frame_idx,
            avg_pre, avg_inf, avg_post, avg_e2e,
            percentile(infer_vec, 50), percentile(infer_vec, 95),
            percentile(e2e_vec, 50), percentile(e2e_vec, 95), percentile(e2e_vec, 99),
            avg_fps, last_det_count
        );
        return 0;
    }

    // ================= CHẾ ĐỘ IMAGE BENCHMARK (CPU PRE HOẶC CUDA PRE) =================
    bool use_cpu_pre = (first_arg == "--cpu-pre");
    int arg_offset = use_cpu_pre ? 1 : 0;

    std::string engine_path = argv[1 + arg_offset];
    std::string image_path  = argv[2 + arg_offset];
    std::string output_path = argv[3 + arg_offset];
    int iterations = (argc >= 5 + arg_offset) ? std::stoi(argv[4 + arg_offset]) : 300;
    std::string json_out = (argc >= 6 + arg_offset) ? argv[5 + arg_offset] : "";

    TRTEngine engine;
    if (!engine.load(engine_path)) return -1;

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    LetterboxInfo info;
    // Warm-up 25 iterations
    for (int i = 0; i < 25; ++i) {
        if (use_cpu_pre) {
            Preprocessor::processCPU(image, engine.getHostInput(), info);
            engine.inferAsyncCPUInput();
            engine.synchronize();
        } else {
            float dp = 0, di = 0;
            engine.inferFusedGPU(image, info, dp, di);
        }
    }

    std::vector<double> pre_vec, infer_vec, post_vec, e2e_vec;
    pre_vec.reserve(iterations);
    infer_vec.reserve(iterations);
    post_vec.reserve(iterations);
    e2e_vec.reserve(iterations);

    std::vector<Detection> detections;

    for (int i = 0; i < iterations; ++i) {
        if (use_cpu_pre) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Preprocessor::processCPU(image, engine.getHostInput(), info);
            auto t1 = std::chrono::high_resolution_clock::now();

            engine.inferAsyncCPUInput();
            engine.synchronize();
            auto t2 = std::chrono::high_resolution_clock::now();

            detections = Postprocessor::process(engine.getHostOutput(), info, 0.25f, 0.45f);
            auto t3 = std::chrono::high_resolution_clock::now();

            double p_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double i_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double n_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            pre_vec.push_back(p_ms);
            infer_vec.push_back(i_ms);
            post_vec.push_back(n_ms);
            e2e_vec.push_back(p_ms + i_ms + n_ms);
        } else {
            float gpu_pre_ms = 0.0f, gpu_infer_ms = 0.0f;
            auto t0 = std::chrono::high_resolution_clock::now();

            engine.inferFusedGPU(image, info, gpu_pre_ms, gpu_infer_ms);

            auto t1 = std::chrono::high_resolution_clock::now();
            detections = Postprocessor::process(engine.getHostOutput(), info, 0.25f, 0.45f);
            auto t2 = std::chrono::high_resolution_clock::now();

            double n_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

            pre_vec.push_back(gpu_pre_ms);
            infer_vec.push_back(gpu_infer_ms);
            post_vec.push_back(n_ms);
            e2e_vec.push_back(total_ms);
        }
    }

    auto mean_of = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    double avg_pre   = mean_of(pre_vec);
    double avg_infer = mean_of(infer_vec);
    double avg_post  = mean_of(post_vec);
    double avg_e2e   = mean_of(e2e_vec);
    double fps       = 1000.0 / std::max(0.001, avg_e2e);

    std::string mode_label = use_cpu_pre ? "CPU_Preprocess" : "CUDA_Fused_Preprocess";

    std::cout << "\n================ NATIVE C++ / TENSORRT BENCHMARK (" << mode_label << ") ================" << std::endl;
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
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Pre-process Latency:    " << avg_pre << " ms" << std::endl;
    std::cout << "Inference Latency:      " << avg_infer << " ms (p50=" << percentile(infer_vec, 50)
              << " ms, p95=" << percentile(infer_vec, 95) << " ms)" << std::endl;
    std::cout << "Post-process (NMS):     " << avg_post << " ms" << std::endl;
    std::cout << "End-to-End Latency:     " << avg_e2e << " ms (p50=" << percentile(e2e_vec, 50)
              << " ms, p95=" << percentile(e2e_vec, 95) << " ms, p99=" << percentile(e2e_vec, 99) << " ms)" << std::endl;
    std::cout << "Throughput (FPS):       " << fps << " frames/s" << std::endl;
    std::cout << "================================================================================\n" << std::endl;

    cv::Mat vis = image.clone();
    drawDetectionsAndHUD(
        vis, detections, fps, avg_e2e, avg_pre, avg_infer, avg_post,
        use_cpu_pre ? "C++ TRT (CPU Pre)" : "C++ TRT (CUDA Pre)"
    );
    cv::imwrite(output_path, vis);

    saveMetricsJson(
        json_out, mode_label, engine_path, iterations,
        avg_pre, avg_infer, avg_post, avg_e2e,
        percentile(infer_vec, 50), percentile(infer_vec, 95),
        percentile(e2e_vec, 50), percentile(e2e_vec, 95), percentile(e2e_vec, 99),
        fps, detections.size()
    );

    return 0;
}
