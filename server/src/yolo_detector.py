from venv import logger
import cv2
import numpy as np
from ray import serve
import logging
import time
import os

@serve.deployment(
    name="YoloDetector",
    num_replicas=1,
    ray_actor_options={"num_cpus": 2} # Metti "num_gpus": 0.5 o 1 quando userai GPU
)
class YoloDetector:
    def __init__(self):
        self.logger = logging.getLogger("ray.serve.detector")
        self.logger.setLevel(logging.INFO)

        self.logger.info("Loading YOLO model")
        base_dir = os.path.dirname(__file__)
        cfg_path = os.path.join(base_dir, "..", "yolo", "yolov3.cfg")
        weights_path = os.path.join(base_dir, "..", "yolo", "yolov3.weights")
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.logger.info("YOLO model ready, output_layers=%d", len(self.output_layers))

    def detect(self, img: np.ndarray):
        start = time.time()
        if img is None:
            self.logger.warning("Received empty image")
            return []
        self.logger.info("Running detection on image shape=%s", img.shape)
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []


        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.3:  # Adjust confidence threshold as needed
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maxima suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        # Prepare results in JSON format
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f'Class {class_ids[i]}'
                confidence = confidences[i]
                results.append({'name': label, 'confidence': confidence, 'xmin': x, 'ymin': y, 'xmax': x + w, 'ymax': y + h})
        elapsed = time.time() - start
        # Log processing time
        self.logger.info("Detection completed in %.4fs, found %d objects", elapsed, len(results))
        return results