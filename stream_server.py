import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Jetson IMX219 Camera</h2><img src='/stream' width='1280'></body></html>")
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except BrokenPipeError:
            pass
        finally:
            cap.release()
    def log_message(self, format, *args):
        pass

print("Stream server running at http://0.0.0.0:8080")
HTTPServer(("0.0.0.0", 8080), MJPEGHandler).serve_forever()
