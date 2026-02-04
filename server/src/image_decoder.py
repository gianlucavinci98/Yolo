import numpy as np
import cv2
from ray import serve

@serve.deployment(
    name="ImageDecoder",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
class ImageDecoder:
    def decode(self, image_bytes: bytes) -> np.ndarray:
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return img