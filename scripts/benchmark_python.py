import argparse
import glob
import json
import time
from pathlib import Path

import torch
import numpy as np
import cv2
from ultralytics import YOLO

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark Ultralytics inference latency and throughput.")
    parser.add_argument("--weights", default=str(repo_root / "models" / "yolov8s.pt"))
    parser.add_argument("--source", help="Image path; defaults to first data/val2017 image or a synthetic 1080p frame.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", default=str(repo_root / "benchmarks" / "pytorch_baseline.json"))
    args = parser.parse_args()
    if args.imgsz < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("--imgsz and --iterations must be positive; --warmup cannot be negative")
    return args


def benchmark(args):
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}. Run scripts/export_onnx.py first or pass --weights.")

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = args.device if args.device.startswith("cuda:") else f"cuda:{args.device}"
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable in this PyTorch environment")

    device_name = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU"
    print(f"Benchmark device: {device_name}")
    model = YOLO(str(weights_path)).to(device)

    source = args.source
    if source:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read benchmark image: {source}")
        source_description = str(Path(source).resolve())
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            candidates.extend(glob.glob(str(repo_root / "data" / "val2017" / extension)))
        if candidates:
            source = sorted(candidates)[0]
            frame = cv2.imread(source)
            source_description = str(Path(source).resolve())
        else:
            frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
            source_description = "synthetic 1080p frame (no validation image found)"

    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        torch.cuda.synchronize(torch.device(device))

    print(f"Warm-up: {args.warmup} iterations")
    for _ in range(args.warmup):
        model.predict(frame, imgsz=args.imgsz, device=device, verbose=False)
    if use_cuda:
        torch.cuda.synchronize(torch.device(device))

    num_iterations = args.iterations
    latencies = []
    print(f"Measuring {num_iterations} iterations...")
    start_total = time.perf_counter()
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, device=device, verbose=False)
        if use_cuda:
            torch.cuda.synchronize(torch.device(device))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_elapsed = time.perf_counter() - start_total
    fps = num_iterations / total_elapsed
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    peak_vram_mb = float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024 * 1024)) if use_cuda else None
    reserved_vram_mb = float(torch.cuda.max_memory_reserved(torch.device(device)) / (1024 * 1024)) if use_cuda else None

    results = {
        "framework": "PyTorch (Python)",
        "precision": "FP32",
        "device": device_name,
        "input_resolution": f"{frame.shape[1]}x{frame.shape[0]} (letterbox to {args.imgsz}x{args.imgsz})",
        "source": source_description,
        "latency_scope": "Ultralytics predict end-to-end, including preprocessing and postprocessing",
        "iterations": num_iterations,
        "throughput_fps": round(fps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "peak_vram_allocated_mb": round(peak_vram_mb, 2) if peak_vram_mb is not None else None,
        "peak_vram_reserved_mb": round(reserved_vram_mb, 2) if reserved_vram_mb is not None else None
    }

    print("\n================ PYTORCH BENCHMARK RESULTS ================")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("===========================================================\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    benchmark(parse_args())
