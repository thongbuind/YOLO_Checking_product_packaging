# import cv2
# import threading
# from utils.visual import FPSCalculator

# class CamThread:
#     def __init__(self, cam_name, url, target_size=(640, 480)):
#         self.cam_name = cam_name
#         self.target_size = target_size
#         self.cap = cv2.VideoCapture(url)
        
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.cap.set(cv2.CAP_PROP_FPS, 60)
        
#         self.frame = None
#         self.running = True
#         self.lock = threading.Lock()
#         self.fps_calc = FPSCalculator()
        
#         self.thread = threading.Thread(target=self.update, daemon=True)
#         self.thread.start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 resized = cv2.resize(frame, self.target_size, 
#                                     interpolation=cv2.INTER_LINEAR)
#                 with self.lock:
#                     self.frame = resized
#                     self.fps_calc.update()

#     def read(self):
#         with self.lock:
#             return (self.frame.copy() if self.frame is not None else None, 
#                     self.fps_calc.get_fps())

#     def release(self):
#         self.running = False
#         self.thread.join(timeout=1.0)
#         self.cap.release()



import cv2
import subprocess
import threading
import numpy as np
from utils.visual import FPSCalculator

class CamThread:
    def __init__(self, cam_name, source, mode="webcam", target_size=(640, 480)):
        self.cam_name = cam_name
        self.source = source
        self.target_size = target_size
        self.mode = mode

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.fps_calc = FPSCalculator()

        if self.mode == "webcam":
            self._init_webcam()
        elif self.mode == "rtsp":
            self._init_rtsp_hw()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _init_webcam(self):
        print(f"[{self.cam_name}] Webcam (AVFoundation)")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)

    def _init_rtsp_hw(self):
        print(f"[{self.cam_name}] RTSP + VideoToolbox decode")

        w, h = self.target_size
        cmd = [
            "ffmpeg",
            "-hwaccel", "videotoolbox",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-vf", f"scale={w}:{h}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame_size = w * h * 3

    def update(self):
        while self.running:

            if self.mode == "webcam":
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, self.target_size)

            elif self.mode == "rtsp":
                raw = self.proc.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    continue

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.target_size[1], self.target_size[0], 3
                )

            with self.lock:
                self.frame = frame
                self.fps_calc.update()

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
