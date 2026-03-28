import cv2
import time
import sys
import numpy as np
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from ultralytics import YOLO
from pathlib import Path

print("Loading TensorRT engine...", flush=True)
model = YOLO("/home/jetson-nx/yolov8n.engine", task="detect")

print("Warming up...", flush=True)
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
for _ in range(3):
    model.predict(dummy, imgsz=640, conf=0.25, verbose=False)
print("Warmup done!", flush=True)

PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
)

print("Opening camera...", flush=True)
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("ERROR: Cannot open camera", flush=True)
    sys.exit(1)
for _ in range(10):
    cap.read()
print("Camera ready!", flush=True)

latest_frame = None
lock = threading.Lock()

def inference_loop():
    global latest_frame
    fps_time = time.monotonic()
    fps_count = 0
    current_fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()
        fps_count += 1
        elapsed = time.monotonic() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.monotonic()
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            latest_frame = jpg.tobytes()

t = threading.Thread(target=inference_loop, daemon=True)
t.start()
while latest_frame is None:
    time.sleep(0.1)
print("Inference running! Server ready.", flush=True)

class YOLOHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = (
                "<html><head><title>Jetson YOLOv8</title></head>"
                "<body style='background:#111;color:#eee;text-align:center'>"
                "<h2>Jetson Orin Nano - YOLOv8n Real-time Detection</h2>"
                "<img src='/stream' width='1280'>"
                "</body></html>"
            )
            self.wfile.write(html.encode())
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    jpg = latest_frame
                if jpg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.033)
        except BrokenPipeError:
            pass
    def log_message(self, format, *args):
        pass

print("http://192.168.219.108:8080", flush=True)
HTTPServer(("0.0.0.0", 8080), YOLOHandler).serve_forever()
