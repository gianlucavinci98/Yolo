from image_decoder import ImageDecoder
from yolo_detector import YoloDetector
from ingress import YoloIngress


decoder = ImageDecoder.bind()
detector = YoloDetector.bind()

# 2. Collega l'Ingress agli altri due passando gli oggetti bindati
# Questi argomenti finiscono nell'__init__ di YoloIngress
app = YoloIngress.bind(decoder, detector)