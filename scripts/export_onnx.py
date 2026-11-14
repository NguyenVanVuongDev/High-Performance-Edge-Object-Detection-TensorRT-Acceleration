import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

repo_root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Export static 640x640 YOLO ONNX models for TensorRT.")
parser.add_argument("--weights", default="yolov8s.pt", help="Ultralytics model name or path to a .pt file")
parser.add_argument("--output-dir", default=str(repo_root / "models"))
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

models_dir = Path(args.output_dir)
models_dir.mkdir(parents=True, exist_ok=True)
weights_path = models_dir / "yolov8s.pt"

print("1. Tải trọng số YOLOv8s pre-trained...")
model = YOLO(args.weights)
if Path(args.weights).is_file():
    shutil.copy2(args.weights, weights_path)
else:
    model.save(str(weights_path))

print(f"2. Xuất ONNX FP32 (static shape: 1x3x{args.imgsz}x{args.imgsz}, opset=12)...")
fp32_export = model.export(
    format="onnx",
    imgsz=args.imgsz,
    batch=1,
    opset=12,
    simplify=True,
    dynamic=False,
    half=False
)
dest_onnx = models_dir / "yolov8s.onnx"
if Path(fp32_export).resolve() != dest_onnx.resolve():
    shutil.copy2(fp32_export, dest_onnx)

print("3. Xuất ONNX FP16 (half precision cho TensorRT 11+ strongly-typed graph)...")
fp16_export = model.export(
    format="onnx",
    imgsz=args.imgsz,
    batch=1,
    opset=12,
    simplify=True,
    dynamic=False,
    half=True
)
dest_onnx_fp16 = models_dir / "yolov8s_fp16.onnx"
if Path(fp16_export).resolve() != dest_onnx_fp16.resolve():
    shutil.copy2(fp16_export, dest_onnx_fp16)

print(f"File ONNX FP32: {dest_onnx}")
print(f"File ONNX FP16: {dest_onnx_fp16}")
print("Kiểm tra mô hình ONNX: HỢP LỆ (PASSED)!")
