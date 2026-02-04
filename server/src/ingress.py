from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import time

@serve.deployment(name="YoloIngress")
class YoloIngress:
    def __init__(self, decoder_handle: DeploymentHandle, detector_handle: DeploymentHandle):
        self.decoder = decoder_handle
        self.detector = detector_handle
        self.logger = logging.getLogger("Yolo-Ingress")
        logging.basicConfig(level=logging.INFO)

    async def __call__(self, http_request: Request):
        form_data = await http_request.form()

        
        if 'frames' not in form_data:
            return JSONResponse({'error': 'No frames found'}, status_code=400)

        image_bytes = await form_data['frames'].read()

        # ASYNCHRONOUS PIPELINE:
        # 1. Ask the decoder to decompress
        ref_decoded = await self.decoder.decode.remote(image_bytes)

        # 2. Pass the reference of the decoded image DIRECTLY to the detector
        # Ray handles data passing in the background (Object Store)
        yolo_start = time.time()
        ref_result = await self.detector.detect.remote(ref_decoded)
        yolo_time = time.time() - yolo_start

        results = []
        results.append({
            'detections': ref_result,
            'yolo_processing_time': yolo_time
        })



        return JSONResponse(results)