import time
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO

def benchmark():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Đo đạc PyTorch Baseline trên: {torch.cuda.get_device_name(0)}")
    
    weights_path = "/kaggle/working/tensorrt-yolo-edge-acceleration/models/yolov8s.pt"
    model = YOLO(weights_path).to(device)

    # Frame 1080p mô phỏng stream camera thực tế
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # 1. Warm-up 100 iterations
    print("Thực hiện 100 iterations warm-up GPU...")
    for _ in range(100):
        _ = model.predict(frame, imgsz=640, verbose=False)
    torch.cuda.synchronize()

    # 2. Benchmark 1.000 iterations
    num_iterations = 1000
    latencies = []
    print(f"Bắt đầu đo {num_iterations} iterations benchmark...")
    
    start_total = time.perf_counter()
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model.predict(frame, imgsz=640, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_elapsed = time.perf_counter() - start_total
    fps = num_iterations / total_elapsed
    
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    results = {
        "framework": "PyTorch (Python)",
        "precision": "FP32",
        "device": torch.cuda.get_device_name(0),
        "input_resolution": "1080p (Letterbox to 640x640)",
        "iterations": num_iterations,
        "throughput_fps": round(fps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }

    print("\n================ PYTORCH BASELINE RESULTS ================")
    print(f"Device:           {results['device']}")
    print(f"Throughput (FPS): {results['throughput_fps']} frames/s")
    print(f"Mean Latency:    {results['mean_latency_ms']} ms")
    print(f"Latency p50:     {results['latency_p50_ms']} ms")
    print(f"Latency p95:     {results['latency_p95_ms']} ms")
    print(f"Peak VRAM:       {results['peak_vram_mb']} MB")
    print("===========================================================\n")

    log_path = "/kaggle/working/tensorrt-yolo-edge-acceleration/scripts/pytorch_baseline.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Đã lưu kết quả đo đạc tại: {log_path}")

if __name__ == "__main__":
    benchmark()
