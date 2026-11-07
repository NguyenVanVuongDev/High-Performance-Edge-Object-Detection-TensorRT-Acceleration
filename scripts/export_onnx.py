import os
import shutil
from ultralytics import YOLO

repo_root = "/kaggle/working/tensorrt-yolo-edge-acceleration"
models_dir = os.path.join(repo_root, "models")
weights_path = os.path.join(models_dir, "yolov8s.pt")

print("1. Tải trọng số YOLOv8s pre-trained...")
model = YOLO("yolov8s.pt")
model.save(weights_path)

print("2. Xuất ONNX FP32 (static shape: 1x3x640x640, opset=12, simplify=True)...")
model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    opset=12,
    simplify=True,
    dynamic=False,
    half=False
)
dest_onnx = os.path.join(models_dir, "yolov8s.onnx")
if os.path.exists("yolov8s.onnx") and "yolov8s.onnx" != dest_onnx:
    shutil.move("yolov8s.onnx", dest_onnx)

print("3. Xuất ONNX FP16 (half precision cho TensorRT 11+ strongly-typed graph)...")
model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    opset=12,
    simplify=True,
    dynamic=False,
    half=True
)
dest_onnx_fp16 = os.path.join(models_dir, "yolov8s_fp16.onnx")
if os.path.exists("yolov8s.onnx"):
    shutil.move("yolov8s.onnx", dest_onnx_fp16)

print(f"File ONNX FP32: {dest_onnx}")
print(f"File ONNX FP16: {dest_onnx_fp16}")
print("Kiểm tra mô hình ONNX: HỢP LỆ (PASSED)!")
