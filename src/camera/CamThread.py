import cv2
import subprocess
import threading
import numpy as np
from utils.visual import FPSCalculator
import time

class CamThread:
    def __init__(self, cam_name, source, mode="webcam", target_size=(640, 480), target_fps=20):
        self.cam_name = cam_name
        self.source = source
        self.target_size = target_size
        self.mode = mode
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.fps_calc = FPSCalculator()
        self.last_capture_time = 0

        if self.mode == "webcam":
            self._init_webcam()
        elif self.mode == "rtsp":
            self._init_rtsp_hw()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _init_webcam(self):
        print(f"[{self.cam_name}] Webcam (AVFoundation) - Target FPS: {self.target_fps}")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)

    def _init_rtsp_hw(self):
        print(f"[{self.cam_name}] RTSP + VideoToolbox decode - Target FPS: {self.target_fps}")

        w, h = self.target_size
        cmd = [
            "ffmpeg",
            "-hwaccel", "videotoolbox",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-vf", f"scale={w}:{h}",
            "-r", str(self.target_fps),  # Giới hạn FPS ở ffmpeg
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame_size = w * h * 3

    def update(self):
        while self.running:
            current_time = time.time()
            
            time_since_last = current_time - self.last_capture_time
            if time_since_last < self.frame_interval:
                sleep_time = self.frame_interval - time_since_last
                time.sleep(max(0.001, sleep_time * 0.5))
                continue

            if self.mode == "webcam":
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue
                frame = cv2.resize(frame, self.target_size)

            elif self.mode == "rtsp":
                raw = self.proc.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    time.sleep(0.001)
                    continue

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.target_size[1], self.target_size[0], 3
                )

            with self.lock:
                self.frame = frame
                self.fps_calc.update()
                self.last_capture_time = current_time

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.fps_calc.get_fps()

    def release(self):
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()
        if hasattr(self, "proc"):
            self.proc.kill()
        print(f"[{self.cam_name}] Released.")
