import numpy as np
import cv2
from ray import serve
import logging


@serve.deployment(
    name="ImageDecoder",
    num_replicas=3,
    ray_actor_options={"num_cpus": 1}
)
class ImageDecoder:
    def __init__(self):
        self.logger = logging.getLogger("ray.serve.decoder")
        self.logger.setLevel(logging.INFO)
        self.logger.info("ImageDecoder initialized")
        pass

    def decode(self, image_bytes: bytes) -> np.ndarray:
        self.logger.info("Decoding image bytes: size=%d", len(image_bytes))
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            self.logger.warning("Failed to decode image (None)")
        else:
            self.logger.info("Decoded image shape=%s", img.shape)
        return img