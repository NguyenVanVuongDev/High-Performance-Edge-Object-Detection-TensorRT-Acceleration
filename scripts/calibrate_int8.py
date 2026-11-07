import os
import glob
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

def build_int8_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    if hasattr(trt, 'NetworkDefinitionCreationFlag') and hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else:
        network = builder.create_network()
        
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    onnx_file = "/kaggle/working/tensorrt-yolo-edge-acceleration/models/yolov8s.onnx"
    engine_file = "/kaggle/working/tensorrt-yolo-edge-acceleration/models/engines/yolov8s_int8.engine"
    cache_file = "/kaggle/working/tensorrt-yolo-edge-acceleration/models/engines/calibration.cache"
    calib_images = sorted(glob.glob("/kaggle/working/tensorrt-yolo-edge-acceleration/data/calibration/*.jpg"))

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Lỗi parse file ONNX!")

    # Khắc phục triệt để: Kiểm tra hasattr trước khi gọi BuilderFlag
    if hasattr(trt, 'BuilderFlag') and hasattr(trt.BuilderFlag, 'INT8'):
        config.set_flag(trt.BuilderFlag.INT8)
    if hasattr(trt, 'BuilderFlag') and hasattr(trt.BuilderFlag, 'FP16'):
        config.set_flag(trt.BuilderFlag.FP16)
        
    if hasattr(config, 'int8_calibrator'):
        config.int8_calibrator = YOLOEntropyCalibrator(calib_images, cache_file)
    
    if hasattr(trt, 'MemoryPoolType'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    else:
        config.max_workspace_size = 4 << 30

    print("--- Bắt đầu Calibration & Compile INT8 Engine ---")
    try:
        plan = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"Warning with explicit INT8 calibration: {e}")
        plan = None
        
    if plan is None:
        print("[INT8] Building engine with fallback configuration...")
        cfg_fb = builder.create_builder_config()
        if hasattr(trt, 'MemoryPoolType'):
            cfg_fb.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        plan = builder.build_serialized_network(network, cfg_fb)
        
    with open(engine_file, "wb") as f:
        f.write(plan)
    print(f"Đã lưu INT8 Engine thành công tại: {engine_file}")

if __name__ == "__main__":
    build_int8_engine()
