import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate detection mAP using Ultralytics validation.")
    parser.add_argument("--model", required=True, help="Path to .pt weights or a compatible TensorRT .engine")
    parser.add_argument("--data", required=True, help="Dataset YAML, for example a COCO or COCO128 YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None, help="Ultralytics device value; defaults to automatic selection")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", help="JSON output path; defaults to benchmarks/map_<model>.json")
    args = parser.parse_args()
    if args.imgsz < 1 or args.batch < 1:
        parser.error("--imgsz and --batch must be positive")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        verbose=False,
    )
    result = {
        "model": str(model_path.resolve()),
        "dataset": str(data_path.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    output_path = Path(args.output) if args.output else repo_root / "benchmarks" / f"map_{model_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()