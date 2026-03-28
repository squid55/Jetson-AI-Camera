"""
Jetson Orin Nano - Fusion Server
Zybo Z7-20 CNN 결과 수신 + 듀얼 MIPI 카메라 YOLOv8 추론 융합

Usage:
    python3 fusion_server.py
    python3 fusion_server.py --config config.yaml
    python3 fusion_server.py --no-display
"""

import argparse
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fusion")


# ──────────────────────────────────────────────
# Zybo CNN 결과 수신기
# ──────────────────────────────────────────────
@dataclass
class CnnResult:
    digit: int = -1
    confidence: float = 0.0
    valid: bool = False
    timestamp_ms: int = 0
    recv_time: float = 0.0  # time.monotonic()


class ZyboReceiver:
    """UDP로 Zybo FPGA CNN 결과 패킷(12B)을 수신."""

    HEADER = 0xAA
    PACKET_SIZE = 12
    STRUCT_FMT = "<BBHIB3x"  # header, digit, confidence, timestamp, status, 3 reserved

    def __init__(self, port: int = 5001, timeout_ms: float = 500):
        self.port = port
        self.timeout_s = timeout_ms / 1000.0
        self._lock = threading.Lock()
        self._latest = CnnResult()
        self._running = False
        self._stats = {"received": 0, "invalid": 0}

    def start(self):
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(1.0)
        t = threading.Thread(target=self._recv_loop, daemon=True, name="zybo-rx")
        t.start()
        log.info("Zybo receiver started on UDP port %d", self.port)

    def stop(self):
        self._running = False
        self._sock.close()

    def get(self) -> CnnResult:
        with self._lock:
            result = CnnResult(
                digit=self._latest.digit,
                confidence=self._latest.confidence,
                valid=self._latest.valid,
                timestamp_ms=self._latest.timestamp_ms,
                recv_time=self._latest.recv_time,
            )
        # 타임아웃 체크
        if result.valid and (time.monotonic() - result.recv_time) > self.timeout_s:
            result.valid = False
        return result

    @property
    def stats(self):
        return self._stats.copy()

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < self.PACKET_SIZE:
                self._stats["invalid"] += 1
                continue

            header, digit, conf_raw, ts, status = struct.unpack(
                self.STRUCT_FMT, data[: self.PACKET_SIZE]
            )

            if header != self.HEADER or digit > 9:
                self._stats["invalid"] += 1
                continue

            self._stats["received"] += 1
            with self._lock:
                self._latest = CnnResult(
                    digit=digit,
                    confidence=conf_raw / 1023.0,
                    valid=bool(status & 0x01),
                    timestamp_ms=ts,
                    recv_time=time.monotonic(),
                )


# ──────────────────────────────────────────────
# GStreamer 카메라 파이프라인
# ──────────────────────────────────────────────
def gst_pipeline(sensor_id: int, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )


# ──────────────────────────────────────────────
# 융합 판단 로직
# ──────────────────────────────────────────────
@dataclass
class FusionResult:
    # YOLO
    yolo_detections: list = field(default_factory=list)
    yolo_fps: float = 0.0
    # Zybo
    zybo_digit: int = -1
    zybo_confidence: float = 0.0
    zybo_valid: bool = False
    # 융합
    fused_labels: list = field(default_factory=list)


def fuse_results(yolo_results, zybo: CnnResult, conf_threshold: float) -> FusionResult:
    """YOLO 탐지 결과와 Zybo CNN 결과를 융합."""
    fusion = FusionResult(
        zybo_digit=zybo.digit,
        zybo_confidence=zybo.confidence,
        zybo_valid=zybo.valid,
    )

    # YOLO 결과 정리
    for r in yolo_results:
        if r.boxes is not None:
            for box in r.boxes:
                fusion.yolo_detections.append({
                    "class": int(box.cls[0]),
                    "name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                })

    # 융합: Zybo 숫자 인식 + YOLO 객체 탐지 결합
    if zybo.valid and zybo.confidence >= conf_threshold:
        fusion.fused_labels.append(f"FPGA-Digit:{zybo.digit} ({zybo.confidence:.0%})")

    for det in fusion.yolo_detections:
        fusion.fused_labels.append(f"{det['name']}:{det['confidence']:.0%}")

    return fusion


# ──────────────────────────────────────────────
# OSD (On-Screen Display)
# ──────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, fusion: FusionResult, cam_id: int) -> np.ndarray:
    """프레임에 탐지 결과 오버레이."""
    h, w = frame.shape[:2]

    # 카메라 ID
    cv2.putText(frame, f"CAM {cam_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # YOLO bbox
    for det in fusion.yolo_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['name']} {det['confidence']:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Zybo CNN 결과 (상단 우측)
    if fusion.zybo_valid:
        zybo_text = f"FPGA Digit: {fusion.zybo_digit} ({fusion.zybo_confidence:.0%})"
        cv2.putText(frame, zybo_text, (w - 350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # FPS (하단 좌측)
    if fusion.yolo_fps > 0:
        cv2.putText(frame, f"FPS: {fusion.yolo_fps:.1f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Jetson AI Camera Fusion Server")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--no-display", action="store_true", help="화면 출력 비활성화")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cam_cfg = cfg["camera"]
    model_cfg = cfg["model"]
    net_cfg = cfg["network"]
    fusion_cfg = cfg["fusion"]

    # YOLO 모델 로드
    from ultralytics import YOLO

    engine_path = Path(model_cfg["engine"])
    if engine_path.exists():
        log.info("Loading TensorRT engine: %s", engine_path)
        model = YOLO(str(engine_path), task=model_cfg["task"])
    else:
        log.warning("Engine not found, using PyTorch weights: %s", model_cfg["weights"])
        model = YOLO(model_cfg["weights"])

    # Zybo 수신기
    zybo = ZyboReceiver(
        port=net_cfg["ports"]["cnn_result"],
        timeout_ms=fusion_cfg["result_timeout_ms"],
    )
    zybo.start()

    # 듀얼 카메라
    s0 = cam_cfg["sensor_0"]
    s1 = cam_cfg["sensor_1"]
    cam0 = cv2.VideoCapture(gst_pipeline(s0["id"], s0["width"], s0["height"], s0["fps"]),
                            cv2.CAP_GSTREAMER)
    cam1 = cv2.VideoCapture(gst_pipeline(s1["id"], s1["width"], s1["height"], s1["fps"]),
                            cv2.CAP_GSTREAMER)

    if not cam0.isOpened() or not cam1.isOpened():
        log.error("Failed to open cameras. Check MIPI connections.")
        return

    log.info("Dual cameras opened. Starting inference loop...")

    frame_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    try:
        while True:
            ret0, frame0 = cam0.read()
            ret1, frame1 = cam1.read()
            if not ret0 or not ret1:
                log.warning("Camera read failed, retrying...")
                continue

            # YOLO batch 추론
            results = model.predict(
                [frame0, frame1],
                imgsz=model_cfg["imgsz"],
                conf=model_cfg["conf_threshold"],
                iou=model_cfg["iou_threshold"],
                batch=model_cfg["batch_size"],
                verbose=False,
            )

            # Zybo 결과 가져오기
            zybo_result = zybo.get()

            # 융합
            fusion0 = fuse_results(results[:1], zybo_result, fusion_cfg["zybo_confidence_threshold"])
            fusion1 = fuse_results(results[1:], zybo_result, fusion_cfg["zybo_confidence_threshold"])

            # FPS 계산
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            fusion0.yolo_fps = current_fps
            fusion1.yolo_fps = current_fps

            # 시각화
            if not args.no_display and fusion_cfg["enable_visualization"]:
                vis0 = draw_overlay(frame0.copy(), fusion0, cam_id=0)
                vis1 = draw_overlay(frame1.copy(), fusion1, cam_id=1)
                combined = np.hstack([vis0, vis1])
                cv2.imshow("Jetson AI Camera - Fusion", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 로그
            if frame_count % fusion_cfg.get("log_interval", 100) == 1:
                zybo_stats = zybo.stats
                log.info(
                    "FPS=%.1f | YOLO dets=%d | Zybo digit=%d conf=%.0f%% valid=%s | rx=%d",
                    current_fps,
                    len(fusion0.yolo_detections) + len(fusion1.yolo_detections),
                    zybo_result.digit,
                    zybo_result.confidence * 100,
                    zybo_result.valid,
                    zybo_stats["received"],
                )

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        zybo.stop()
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
