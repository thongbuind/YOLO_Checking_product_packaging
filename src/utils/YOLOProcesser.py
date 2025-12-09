import numpy as np
import threading
from queue import Queue, Full, Empty
import json
import time
from pathlib import Path
import traceback

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
config_file = project_root / "config" / "config.json"

with open(config_file, 'r') as f:
    config = json.load(f)
classes = config["classes"]

class YOLOProcessor:
    def __init__(self, model, image_size, conf_threshold=0.7, max_fps=10):
        self.model = model
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.max_fps = max_fps
        self.last_submit_time = 0

        self.input_queue = Queue(maxsize=4)
        self.output_queue = Queue(maxsize=4)

        self.running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def process(self):
        while self.running:
            try:
                batch_data = self.input_queue.get(timeout=0.1)
                
                if batch_data is None:
                    if not self.running:
                        break
                    continue

                cam_names, frames = batch_data

                # Chạy YOLO inference
                results = self.model(frames, imgsz=self.image_size, 
                                   conf=self.conf_threshold, verbose=False)

                processed = {}

                for cam_name, result in zip(cam_names, results):
                    slot_boxes = []
                    item_boxes = []

                    # Xử lý OBB results
                    if result.obb is not None and len(result.obb) > 0:
                        for b in result.obb:
                            cls_id = int(b.cls[0])
    
                            # Lấy nguyên 4 điểm chéo (8 số)
                            points = b.xyxyxyxy[0].cpu().numpy().reshape(4, 2)  # shape: (4, 2)

                            if cls_id == 0:  # slot
                                slot_boxes.append(points)
                            elif 1 <= cls_id < len(classes):
                                item_name = classes[cls_id]
                                item_boxes.append((item_name, points))

                    processed[cam_name] = {
                        "slot_boxes": slot_boxes,
                        "item_boxes": item_boxes
                    }

                # Đẩy kết quả vào output queue
                self.output_queue.put(processed)

            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] YOLO processing failed: {e}")
                traceback.print_exc()
                try:
                    self.output_queue.put(None, block=False)
                except Full:
                    pass

    def submit(self, cam_names, frames):
        """
        Submit batch frames để xử lý (non-blocking)
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_submit_time < 1.0 / self.max_fps:
            return False
        
        # Nếu queue đầy, bỏ qua frame cũ
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except:
                pass
        
        try:
            self.input_queue.put((cam_names, frames), block=False)
            self.last_submit_time = current_time
            return True
        except Full:
            return False

    def get_results(self):
        """
        Lấy kết quả detection (non-blocking)
        """
        try:
            return self.output_queue.get(block=False)
        except Empty:
            return None

    def stop(self):
        """
        Dừng YOLO processor
        """
        self.running = False
        try:
            self.input_queue.put(None, block=False)
        except Full:
            pass
        self.thread.join(timeout=2.0)
        print("[INFO] YOLO Processor stopped")
