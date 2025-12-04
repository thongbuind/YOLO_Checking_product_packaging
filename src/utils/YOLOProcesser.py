import threading
from queue import Queue
import json
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
config_file = project_root / "config.json"

with open(config_file, 'r') as f:
    config = json.load(f)
classes = config["classes"]

class YOLOProcessor:
    def __init__(self, model, image_size, conf_threshold=0.7):
        self.model = model
        self.image_size = image_size
        self.conf_threshold = conf_threshold

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
                    continue

                cam_names, frames = batch_data

                results = self.model(frames, imgsz=self.image_size, conf=self.conf_threshold, verbose=False)

                processed = {}

                for cam_name, result in zip(cam_names, results):

                    slot_boxes = []
                    item_boxes = []

                    if result.boxes is not None and len(result.boxes) > 0:

                        for b in result.boxes:
                            cls_id = int(b.cls[0])
                            xyxy = b.xyxy[0].cpu().numpy()

                            if cls_id == 0:
                                slot_boxes.append(xyxy)

                            elif 1 <= cls_id < len(classes):
                                item_name = classes[cls_id]
                                item_boxes.append((item_name, xyxy))

                    processed[cam_name] = {
                        "slot_boxes": slot_boxes,
                        "item_boxes": item_boxes
                    }

                self.output_queue.put(processed)

            except Exception as e:
                print(f"YOLO processing error: {e}")


    def submit(self, cam_names, frames):
        if not self.input_queue.full():
            self.input_queue.put((cam_names, frames))

    def get_results(self):
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
