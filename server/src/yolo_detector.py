import cv2
import numpy as np
from ray import serve

@serve.deployment(
    name="YoloDetector",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1} # Metti "num_gpus": 0.5 o 1 quando userai GPU
)
class YoloDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet('../yolo/yolov3.cfg', '../yolo/yolov3.weights')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, img: np.ndarray):
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
        
        return results