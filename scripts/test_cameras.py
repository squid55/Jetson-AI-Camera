"""
듀얼 MIPI 카메라 테스트 스크립트
카메라 연결 확인 + 프레임 캡처 + FPS 측정

Usage:
    python3 scripts/test_cameras.py
    python3 scripts/test_cameras.py --single 0
    python3 scripts/test_cameras.py --save-snapshot
"""

import argparse
import time

import cv2


def gst_pipeline(sensor_id: int, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )


def test_single(sensor_id: int, save: bool = False):
    """단일 카메라 테스트."""
    print(f"[CAM {sensor_id}] Opening...")
    cap = cv2.VideoCapture(gst_pipeline(sensor_id), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"[CAM {sensor_id}] FAILED to open")
        return False

    # 워밍업
    for _ in range(10):
        cap.read()

    # FPS 측정
    count = 0
    start = time.monotonic()
    while count < 60:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

    elapsed = time.monotonic() - start
    fps = count / elapsed if elapsed > 0 else 0

    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"[CAM {sensor_id}] OK - {w}x{h} @ {fps:.1f} FPS")
        if save:
            path = f"cam{sensor_id}_snapshot.jpg"
            cv2.imwrite(path, frame)
            print(f"[CAM {sensor_id}] Snapshot saved: {path}")
    else:
        print(f"[CAM {sensor_id}] Failed to read frame")

    cap.release()
    return ret


def test_dual(save: bool = False):
    """듀얼 카메라 동시 캡처 테스트."""
    print("Opening dual cameras...")
    cam0 = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
    cam1 = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)

    if not cam0.isOpened() or not cam1.isOpened():
        print("FAILED: Could not open both cameras")
        cam0.release()
        cam1.release()
        return

    # FPS 측정
    count = 0
    start = time.monotonic()
    while count < 60:
        ret0, f0 = cam0.read()
        ret1, f1 = cam1.read()
        if not ret0 or not ret1:
            break
        count += 1

    elapsed = time.monotonic() - start
    fps = count / elapsed if elapsed > 0 else 0
    print(f"Dual camera FPS: {fps:.1f}")

    if save:
        ret0, f0 = cam0.read()
        ret1, f1 = cam1.read()
        if ret0:
            cv2.imwrite("cam0_snapshot.jpg", f0)
        if ret1:
            cv2.imwrite("cam1_snapshot.jpg", f1)
        print("Snapshots saved")

    cam0.release()
    cam1.release()


def main():
    parser = argparse.ArgumentParser(description="MIPI Camera Test")
    parser.add_argument("--single", type=int, default=-1, help="단일 카메라 테스트 (0 or 1)")
    parser.add_argument("--save-snapshot", action="store_true", help="스냅샷 저장")
    args = parser.parse_args()

    if args.single >= 0:
        test_single(args.single, args.save_snapshot)
    else:
        print("=== Single Camera Tests ===")
        test_single(0, args.save_snapshot)
        test_single(1, args.save_snapshot)
        print("\n=== Dual Camera Test ===")
        test_dual(args.save_snapshot)


if __name__ == "__main__":
    main()
