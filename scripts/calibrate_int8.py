import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import tensorrt as trt

class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_images, cache_file, shape=(1, 3, 640, 640)):
        super().__init__()
        self.calib_images = calib_images
        self.cache_file = cache_file
        self.shape = shape
        self.batch_size = shape[0]
        self.current_idx = 0
        
        # Cấp phát CUDA memory bằng PyTorch có sẵn trên Kaggle
        self.device_buffer = torch.zeros(shape, dtype=torch.float32, device="cuda")
        print(f"[Calibrator] Khởi tạo Calibrator với {len(calib_images)} ảnh (PyTorch CUDA buffer).")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.calib_images):
            print("[Calibrator] Hoàn thành toàn bộ batch calibration.")
            return None
        
        img_path = self.calib_images[self.current_idx]
        self.current_idx += 1
        
        # Letterbox 640x640 chuẩn YOLO
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        dw, dh = (640 - nw) // 2, (640 - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        
        # BGR -> RGB, HWC -> CHW, /255.0
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        batch = np.ascontiguousarray(np.expand_dims(chw, axis=0))
        
        # Copy vào PyTorch CUDA tensor
        self.device_buffer.copy_(torch.from_numpy(batch))
        
        # Trả về memory pointer trên GPU
        return [int(self.device_buffer.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Đọc cache calibration từ: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[Calibrator] Lưu cache calibration vào: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_file, engine_file, cache_file, calibration_dir, max_images, workspace_gb):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    if hasattr(trt, 'NetworkDefinitionCreationFlag') and hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else:
        network = builder.create_network()
        
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    calib_images = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        calib_images.extend(glob.glob(os.path.join(calibration_dir, extension)))
    calib_images = sorted(calib_images)[:max_images]
    if not calib_images:
        raise FileNotFoundError(f"No calibration images found in: {calibration_dir}")
    if not os.path.isfile(onnx_file):
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Lỗi parse file ONNX!")

    if not hasattr(trt, "BuilderFlag") or not hasattr(trt.BuilderFlag, "INT8"):
        raise RuntimeError("This TensorRT installation does not expose INT8 builder support")
    config.set_flag(trt.BuilderFlag.INT8)
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    if not hasattr(config, "int8_calibrator"):
        raise RuntimeError("This TensorRT installation does not support Python INT8 calibrators")
    config.int8_calibrator = YOLOEntropyCalibrator(calib_images, cache_file)

    if hasattr(trt, 'MemoryPoolType'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    else:
        config.max_workspace_size = int(workspace_gb * (1 << 30))

    print(f"--- Building INT8 engine with {len(calib_images)} calibration images ---")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine; see TensorRT log above")

    os.makedirs(os.path.dirname(os.path.abspath(engine_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    with open(engine_file, "wb") as f:
        f.write(plan)
    print(f"Đã lưu INT8 Engine thành công tại: {engine_file}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a TensorRT INT8 engine using representative images.")
    parser.add_argument("--onnx", default=str(repo_root / "models" / "yolov8s.onnx"))
    parser.add_argument("--calibration-dir", default=str(repo_root / "data" / "calibration"))
    parser.add_argument("--output", default=str(repo_root / "models" / "engines" / "yolov8s_int8.engine"))
    parser.add_argument("--cache", default=str(repo_root / "models" / "engines" / "calibration.cache"))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    args = parser.parse_args()
    if args.max_images < 1 or args.workspace_gb <= 0:
        parser.error("--max-images and --workspace-gb must be positive")
    build_int8_engine(args.onnx, args.output, args.cache, args.calibration_dir, args.max_images, args.workspace_gb)
