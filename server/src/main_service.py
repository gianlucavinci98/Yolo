from image_decoder import ImageDecoder
from yolo_detector import YoloDetector
from ingress import YoloIngress


decoder = ImageDecoder.bind()
detector = YoloDetector.bind()

# 2. Connect the Ingress to the other two by passing the bound objects
# These arguments end up in the __init__ of YoloIngress
app = YoloIngress.bind(decoder, detector)