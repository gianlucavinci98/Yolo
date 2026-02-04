from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import time

module_logger = logging.getLogger("ray.serve.ingress")

@serve.deployment(name="YoloIngress")
class YoloIngress:
    def __init__(self, decoder_handle: DeploymentHandle, detector_handle: DeploymentHandle):
        self.decoder = decoder_handle
        self.detector = detector_handle
        
        self.logger = module_logger
        self.logger.setLevel(logging.INFO)
        self.logger.info("Ingress initialized - Logger configurato correttemente")

    async def __call__(self, http_request: Request):
        self.logger.info(
            "Incoming request: method=%s content_type=%s",
            http_request.method,
            http_request.headers.get("content-type")
        )
        form_data = await http_request.form()

        
        if 'frames' not in form_data:
            self.logger.warning("No frames found in form data. keys=%s", list(form_data.keys()))
            return JSONResponse({'error': 'No frames found'}, status_code=400)

        image_bytes = await form_data['frames'].read()
        self.logger.info("Received frame bytes: size=%d", len(image_bytes))
        # ASYNCHRONOUS PIPELINE:
        # 1. Ask the decoder to decompress
        ref_decoded = await self.decoder.decode.remote(image_bytes)

        # 2. Pass the reference of the decoded image DIRECTLY to the detector
        # Ray handles data passing in the background (Object Store)
        yolo_start = time.time()
        ref_result = await self.detector.detect.remote(ref_decoded)
        yolo_time = time.time() - yolo_start
        self.logger.info("Detection done in %.4fs", yolo_time)

        results = []
        results.append({
            'detections': ref_result,
            'yolo_processing_time': yolo_time
        })



        return JSONResponse(results)