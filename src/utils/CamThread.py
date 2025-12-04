import cv2
import threading
from utils.visual import FPSCalculator

class CamThread:
    def __init__(self, cam_name, url, target_size=(640, 480)):
        self.cam_name = cam_name
        self.target_size = target_size
        self.cap = cv2.VideoCapture(url)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.fps_calc = FPSCalculator()
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                resized = cv2.resize(frame, self.target_size, 
                                    interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.frame = resized
                    self.fps_calc.update()

    def read(self):
        with self.lock:
            return (self.frame.copy() if self.frame is not None else None, 
                    self.fps_calc.get_fps())

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
