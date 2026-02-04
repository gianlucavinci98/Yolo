import numpy as np
import cv2
from ray import serve
import logging


@serve.deployment(
    name="ImageDecoder",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
class ImageDecoder:
    def __init__(self):
        # logger.info("ImageDecoder initialized")
        pass

    def decode(self, image_bytes: bytes) -> np.ndarray:
        # logger.info("Decoding image bytes: size=%d", len(image_bytes))
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # if img is None:
        #     logger.warning("Failed to decode image (None)")
        # else:
        #     logger.info("Decoded image shape=%s", img.shape)
        # return img
        return img