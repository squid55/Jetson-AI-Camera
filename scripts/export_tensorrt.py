"""
YOLOv8n -> TensorRT INT8 엔진 변환 스크립트
Jetson Orin Nano 4GB 최적화

Usage:
    python3 scripts/export_tensorrt.py
    python3 scripts/export_tensorrt.py --model yolov8n.pt --imgsz 640
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Jetson 환경 확인."""
    checks = []

    # TensorRT
    try:
        import tensorrt
        checks.append(f"TensorRT: {tensorrt.__version__}")
    except ImportError:
        checks.append("TensorRT: NOT FOUND")
        print("ERROR: TensorRT not installed. JetPack SDK required.")
        sys.exit(1)

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(f"CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})")
        else:
            checks.append("CUDA: not available")
    except ImportError:
        checks.append("PyTorch: NOT FOUND")

    # ultralytics
    try:
        import ultralytics
        checks.append(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    print("=== Environment ===")
    for c in checks:
        print(f"  {c}")
    print()


def export_engine(model_path: str, imgsz: int, batch: int, workspace: int):
    """TensorRT INT8 엔진 변환."""
    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Exporting to TensorRT INT8 (imgsz={imgsz}, batch={batch}, workspace={workspace}GB)...")
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=True,
        int8=True,
        batch=batch,
        workspace=workspace,
        device=0,
    )

    print(f"\nExport complete: {engine_path}")
    print(f"Engine size: {Path(engine_path).stat().st_size / 1024 / 1024:.1f} MB")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 TensorRT Export for Jetson")
    parser.add_argument("--model", default="yolov8n.pt", help="PyTorch 모델 경로")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=2, help="배치 크기 (듀얼 카메라=2)")
    parser.add_argument("--workspace", type=int, default=1, help="TensorRT workspace (GB)")
    args = parser.parse_args()

    check_environment()
    export_engine(args.model, args.imgsz, args.batch, args.workspace)


if __name__ == "__main__":
    main()
